import argparse
import asyncio
import json
import logging
import os
import random
import sys
from datetime import timedelta
from multiprocessing import Process, SimpleQueue
from time import time

import orjsonl
import requests
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)
from tqdm import tqdm

sys.path.insert(0, "./")
from pipelines.utils import get_logger
from retrieval.milvus_index import MilvusIndex
from tasks.defaults import DEFAULT_QDRANT_COLLECTION_NAME

# Use Qdrant collection name as default for Milvus too
DEFAULT_MILVUS_COLLECTION_NAME = DEFAULT_QDRANT_COLLECTION_NAME

logger = get_logger(__name__)


def embed_tei(text, is_query, query_prefix, ports):
    """Embed text using the embedding model server."""
    if not isinstance(text, list):
        text = [text]
    if is_query:
        text = [query_prefix + t for t in text]
    
    # Check if ports are available
    if not ports or len(ports) == 0:
        logger.error("No model port provided. Use --model_port to specify embedding model port(s).")
        return None
        
    selected_port = random.choice(ports)  # load balancing
    resp = requests.post(
        f"http://0.0.0.0:{selected_port}/embed",
        json={"inputs": text, "normalize": True, "truncate": True},
        timeout=300,  # seconds
    )
    if resp.status_code == 413:
        # the batch has been too long for HTTP
        logging.warning(
            "Skipping batch because it was too large for HTTP. Consider decreasing --embedding_batch_size"
        )
        return None
    if resp.status_code != 200:
        logging.warning(
            "Skipping batch because the embedding server returned error code %s",
            str(resp.status_code),
        )
        return None
    embedding = resp.json()
    return embedding


def index_batch(input_queue, output_queue, model_ports, query_prefix):
    """Process batches of blocks for indexing."""
    while True:
        item = input_queue.get()
        if item is None:
            break
        batch_blocks = item

        batch_text = [
            "Title: "
            + block.get("full_section_title", "")
            + ". "
            + block.get("content", "").strip()
            for block in batch_blocks
        ]
        batch_embeddings = embed_tei(
            list(batch_text), is_query=False, query_prefix=query_prefix, ports=model_ports
        )  # Ensure list is passed for batch processing

        if batch_embeddings is not None:
            output_queue.put((batch_blocks, batch_embeddings))

    output_queue.put(None)


def commit_to_index(
    num_workers,
    input_queue,
    collection_name,
    collection_size,
    embedding_dim,
    uri=None,
    token=None,
):
    """Create and populate Milvus collection with documents."""
    pbar = tqdm(
        desc="Indexing collection",
        miniters=1e-6,
        unit_scale=1,
        unit=" Blocks",
        dynamic_ncols=True,
        smoothing=0,
        total=collection_size,
    )

    # Connect to Milvus - either local server or Zilliz Cloud
    if uri and token:
        logger.info(f"Connecting to Zilliz Cloud via URI: {uri}")
        connections.connect(
            "default", 
            uri=uri,
            token=token,
            secure=True
        )
    else:
        logger.info("Connecting to local Milvus server at localhost:19530")
        connections.connect("default", host="localhost", port="19530")
    
    # Check if collection exists, if not create it
    if not utility.has_collection(collection_name):
        logger.info(
            "Did not find collection %s in Milvus, creating it...",
            collection_name,
        )
        
        # Define fields for the collection
        id_field = FieldSchema(
            name="id", 
            dtype=DataType.INT64, 
            is_primary=True
        )
        
        vector_field = FieldSchema(
            name="embedding", 
            dtype=DataType.FLOAT_VECTOR, 
            dim=embedding_dim
        )
        
        text_field = FieldSchema(
            name="text", 
            dtype=DataType.VARCHAR, 
            max_length=65535
        )
        
        title_field = FieldSchema(
            name="title", 
            dtype=DataType.VARCHAR, 
            max_length=256
        )
        
        full_section_title_field = FieldSchema(
            name="full_section_title", 
            dtype=DataType.VARCHAR, 
            max_length=512
        )
        
        language_field = FieldSchema(
            name="language", 
            dtype=DataType.VARCHAR, 
            max_length=16
        )
        
        block_type_field = FieldSchema(
            name="block_type", 
            dtype=DataType.VARCHAR, 
            max_length=32
        )
        
        last_edit_date_field = FieldSchema(
            name="last_edit_date", 
            dtype=DataType.VARCHAR, 
            max_length=32
        )
        
        # Create collection schema
        schema = CollectionSchema(
            fields=[
                id_field, 
                vector_field, 
                text_field, 
                title_field, 
                full_section_title_field, 
                language_field, 
                block_type_field, 
                last_edit_date_field
            ],
            description="Text blocks with embeddings"
        )
        
        # Create collection
        collection = Collection(
            name=collection_name, 
            schema=schema,
            shards_num=2
        )
        
        # Create index on vector field
        index_params = {
            "metric_type": "IP",  # Inner Product (for DOT similarity)
            "index_type": "HNSW",
            "params": {
                "M": 64,
                "efConstruction": 100
            }
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("Collection creation was successful")
    else:
        collection = Collection(collection_name)
        collection.load()

    finished_workers = 0
    while True:
        item = input_queue.get()
        if item is None:
            finished_workers += 1
            if finished_workers == num_workers:
                break
            continue
        
        batch_blocks, batch_embeddings = item
        
        # Prepare data for insertion
        ids = [
            abs(
                hash(
                    block.get("content", "")
                    + " "
                    + block.get("full_section_title", "")
                    + block.get("language", "")
                )
            )
            for block in batch_blocks
        ]
        
        # Prepare entities for insertion
        entities = [
            ids,  # id field
            batch_embeddings,  # embedding field
            [block.get("content", "") for block in batch_blocks],  # text field
            [block.get("document_title", "") for block in batch_blocks],  # title field
            [block.get("full_section_title", "") for block in batch_blocks],  # full_section_title field
            [block.get("language", "") for block in batch_blocks],  # language field
            [block.get("block_type", "") for block in batch_blocks],  # block_type field
            [block.get("last_edit_date", "") for block in batch_blocks],  # last_edit_date field
        ]
        
        # Insert data
        try:
            collection.insert(entities)
            pbar.update(len(batch_blocks))
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
    
    # Flush the collection to ensure all data is persisted
    collection.flush()
    
    # Release collection and disconnect
    utility.index_building_progress(collection_name)
    connections.disconnect("default")


def batch_generator(collection_file, embedding_batch_size):
    """Generator function to yield batches of data from the queue."""
    batch = []
    for block in orjsonl.stream(collection_file):
        batch.append(block)
        if len(batch) == embedding_batch_size:
            yield batch
            batch = []

    # yield the last partial batch
    if len(batch) > 0:
        yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_file", type=str, default=None, help=".jsonl file to read from."
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        choices=MilvusIndex.get_supported_embedding_models(),
        default="BAAI/bge-m3",
    )
    parser.add_argument(
        "--model_port",
        type=int,
        nargs="+",
        default=None,
        help="The port(s) to which the embedding model is accessible. In multi-GPU settings, you can run the embedding server on different GPUs and ports.",
    )
    parser.add_argument(
        "--num_embedding_workers",
        type=int,
        default=10,
        help="The number of processes that send embedding requests to GPU. Using too few will underutilize the GPU, and using too many will add overhead.",
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=48,
        help="The size of each request sent to GPU. The actual batch size is `embedding_batch_size * num_embedding_workers`",
    )
    parser.add_argument(
        "--collection_name", default=DEFAULT_MILVUS_COLLECTION_NAME, type=str
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="If set, will index the provided `--collection_file`.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, will run a test query on the provided index.",
    )
    parser.add_argument(
        "--zilliz_uri",
        type=str,
        default=None,
        help="URI for connecting to Zilliz Cloud. If provided, will use Zilliz Cloud instead of local Milvus.",
    )
    parser.add_argument(
        "--zilliz_token",
        type=str,
        default=None,
        help="API token for connecting to Zilliz Cloud. Required if --zilliz_uri is provided.",
    )

    args = parser.parse_args()

    # Get embedding model parameters
    embedding_params = MilvusIndex.get_embedding_model_parameters(
        args.embedding_model_name
    )
    embedding_size = embedding_params["embedding_dimension"]
    query_prefix = embedding_params["query_prefix"]
    
    logger.info(f"Model ports: {args.model_port}")

    if args.index:
        collection_size = 0
        size_file = os.path.join(
            os.path.dirname(args.collection_file), "collection_size.txt"
        )
        try:
            with open(size_file) as f:
                collection_size = int(f.read().strip())
        except Exception as e:
            logger.warning(
                "Could not read the collection size from %s, defaulting to zero.",
                size_file,
            )

        input_queue = SimpleQueue()
        vector_queue = SimpleQueue()
        start_time = time()

        milvus_worker = Process(
            target=commit_to_index,
            args=(
                args.num_embedding_workers,
                vector_queue,
                args.collection_name,
                collection_size,
                embedding_size,
                args.zilliz_uri,
                args.zilliz_token,
            ),
        )
        all_workers = [milvus_worker]

        for _ in range(args.num_embedding_workers):
            p = Process(
                target=index_batch, 
                args=(input_queue, vector_queue, args.model_port, query_prefix)
            )
            all_workers.append(p)

        for p in all_workers:
            p.start()

        # main process reads and feeds the collection to workers
        batches = batch_generator(args.collection_file, args.embedding_batch_size)
        for batch in batches:
            input_queue.put(batch)
        for _ in range(args.num_embedding_workers):
            input_queue.put(None)

        for p in all_workers:
            p.join()

        end_time = time()
        logger.info(
            "Indexing took %s", str(timedelta(seconds=int(end_time - start_time)))
        )

    if args.test:
        # Retrieve a test query
        logger.info("Testing the index")
        queries = [
            "Tell me about Haruki Murakami",
            "Who is the current monarch of the UK?",
        ]

        with MilvusIndex(
            embedding_model_name=args.embedding_model_name,
            collection_name=args.collection_name,
            uri=args.zilliz_uri,
            token=args.zilliz_token,
            use_onnx=True
        ) as index:
            results = asyncio.run(index.search(queries, 5))
            # Convert pydantic models to dict for JSON serialization
            results_dict = [result.dict() for result in results]
            logger.info(json.dumps(results_dict, indent=2, ensure_ascii=False))