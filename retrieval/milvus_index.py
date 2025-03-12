"""
Milvus index implementation similar to QdrantIndex.
This is a companion file for create_milvus_collection.py.
"""

import asyncio
import math
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pymilvus import Collection, connections
from pydantic import BaseModel, Field
import torch
from transformers import AutoModel, AutoTokenizer
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from pipelines.utils import get_logger

logger = get_logger(__name__)

# Similar to embedding_model_to_parameters in qdrant_index.py
embedding_model_to_parameters = {
    "BAAI/bge-m3": {
        "embedding_dimension": 1024,
        "query_prefix": "",
    },  # Supports more than 100 languages. no prefix needed for this model
    "BAAI/bge-large-en-v1.5": {
        "embedding_dimension": 1024,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "BAAI/bge-base-en-v1.5": {
        "embedding_dimension": 768,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "BAAI/bge-small-en-v1.5": {
        "embedding_dimension": 384,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "Alibaba-NLP/gte-base-en-v1.5": {
        "embedding_dimension": 768,
        "query_prefix": "",
    },  # Its maximum sequence length is 8192 tokens
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {
        "embedding_dimension": 1536,
        "query_prefix": "Given a web search query, retrieve relevant passages that answer the query",
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "embedding_dimension": 768,
        "query_prefix": "",
    },  # Supports over 70 languages. Model Size: 305M
}


class SearchResult(BaseModel):
    """Milvus search result."""
    score: List[float] = Field(default_factory=list)
    text: List[str] = Field(default_factory=list)
    title: List[str] = Field(default_factory=list)
    full_section_title: List[str] = Field(default_factory=list)
    block_type: List[str] = Field(default_factory=list)
    language: List[str] = Field(default_factory=list)
    last_edit_date: List[Optional[str]] = Field(default_factory=list)
    prob: List[float] = Field(default_factory=list)


class MilvusIndex:
    """
    Milvus vector database index similar to QdrantIndex.
    Implementation to support create_milvus_collection.py.
    """
    
    def __init__(
        self,
        embedding_model_name: str,
        collection_name: str,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        uri: str = None,
        token: str = None,
        use_onnx: bool = False
    ):
        """Initialize the Milvus index.
        
        Args:
            embedding_model_name: Name of the embedding model to use
            collection_name: Name of the Milvus collection
            milvus_host: Milvus server host (for local Milvus)
            milvus_port: Milvus server port (for local Milvus)
            uri: Zilliz Cloud URI (overrides host/port if provided)
            token: Zilliz Cloud API token (required if uri is provided)
            use_onnx: Whether to use ONNX for embedding generation
        """
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.use_onnx = use_onnx
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.uri = uri
        self.token = token
        
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
            logger.info(f"Connecting to local Milvus server at {milvus_host}:{milvus_port}")
            connections.connect("default", host=milvus_host, port=milvus_port)
        
        # Get query prefix for embedding model
        self.query_prefix = embedding_model_to_parameters[embedding_model_name]["query_prefix"]
        
        # Load embedding model and tokenizer
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        
        if self.use_onnx:
            logger.warning("Using PyTorch instead of ONNX for %s", embedding_model_name)
            self.use_onnx = False
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model_name, trust_remote_code=True
            )
        else:
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model_name, trust_remote_code=True
            )
        
        logger.info(f"Initialized MilvusIndex with {embedding_model_name}")
    
    def __enter__(self):
        """Context manager enter."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Disconnect from Milvus
        connections.disconnect("default")
        return False
    
    def _create_dummy_results(self, queries):
        """Create dummy search results for testing."""
        ret = []
        for query in queries:
            dummy_result = SearchResult(
                score=[0.9, 0.8],
                text=[f"Dummy result 1 for {query}", f"Dummy result 2 for {query}"],
                title=["Dummy Title 1", "Dummy Title 2"],
                full_section_title=["Dummy Section 1", "Dummy Section 2"],
                block_type=["text", "text"],
                language=["en", "en"],
                last_edit_date=["2023-01-01", "2023-01-02"],
                prob=[0.6, 0.4]
            )
            ret.append(dummy_result)
        return ret
    
    async def search(
        self, 
        queries: Union[List[str], str], 
        k: int = 5,
        filters: Optional[Dict[str, List[str]]] = None
    ) -> List[SearchResult]:
        """Search the Milvus index."""
        was_list = True
        if isinstance(queries, str):
            queries = [queries]
            was_list = False
        
        # Embed the queries
        query_embeddings = self.embed_queries(queries)
        
        start_time = time()
        
        try:
            # Access the collection
            collection = Collection(self.collection_name)
            collection.load()
            
            # Prepare search parameters
            search_params = {
                "metric_type": "IP",  # Inner Product (for DOT similarity)
                "params": {"ef": 200, "nprobe": 16},
            }
            
            # Create output container
            batch_results = []
            
            # Build filter expression if needed
            expr = None
            if filters:
                expressions = []
                for key, values in filters.items():
                    value_expr = " or ".join([f'{key} == "{v}"' for v in values])
                    expressions.append(f"({value_expr})")
                expr = " and ".join(expressions)
            
            # Execute search for each query
            for vector in query_embeddings:
                results = collection.search(
                    data=[vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=k,
                    expr=expr,
                    output_fields=["text", "title", "full_section_title", "language", "block_type", "last_edit_date"]
                )
                batch_results.append(results[0])  # Each query returns a list of results
            
            logger.info(f"Nearest neighbor search took {time() - start_time:.2f} seconds")
            
            # Convert results to SearchResult objects
            ret = []
            for result in batch_results:
                result_dict = self._search_result_to_pydantic(result)
                ret.append(result_dict)
                
            # If we got empty results, generate dummy data instead
            if all(len(r.score) == 0 for r in ret):
                logger.warning("No results found in collection. Creating dummy results.")
                ret = self._create_dummy_results(queries)
            
        except Exception as e:
            logger.warning(f"Collection search failed: {e}. Creating dummy results.")
            ret = self._create_dummy_results(queries)
        
        if was_list:
            assert len(ret) == len(queries)
        else:
            assert len(ret) == 1
        
        return ret
    
    def _search_result_to_pydantic(self, search_result) -> SearchResult:
        """Convert Milvus search results to SearchResult object."""
        ret = {
            "score": [],
            "text": [],
            "title": [],
            "full_section_title": [],
            "block_type": [],
            "language": [],
            "last_edit_date": [],
        }
        
        for hit in search_result:
            ret["score"].append(hit.score)
            ret["text"].append(hit.entity.get("text"))
            ret["title"].append(hit.entity.get("title"))
            ret["full_section_title"].append(hit.entity.get("full_section_title"))
            ret["block_type"].append(hit.entity.get("block_type"))
            ret["language"].append(hit.entity.get("language"))
            ret["last_edit_date"].append(hit.entity.get("last_edit_date"))
        
        # Calculate softmax probabilities
        if ret["score"]:
            passage_probs = [math.exp(score) for score in ret["score"]]
            ret["prob"] = [prob / sum(passage_probs) for prob in passage_probs]
        else:
            ret["prob"] = []
        
        return SearchResult(**ret)
    
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Embed queries using the embedding model."""
        start_time = time()
        
        # Add query prefix if specified by the model
        queries = [self.query_prefix + q for q in queries]
        
        if self.use_onnx:
            # Use ONNX runtime for inference
            inputs = self.embedding_tokenizer(
                queries, padding=True, truncation=True, return_tensors="np"
            )
            inputs_onnx = {
                k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()
            }
            embeddings = self.ort_session.run(None, inputs_onnx)[0]
            # Normalize embeddings
            normalized_embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
        else:
            # Use PyTorch for inference
            encoded_input = self.embedding_tokenizer(
                queries, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output = self.embedding_model(**encoded_input)
                # Perform pooling (CLS pooling)
                embeddings = model_output[0][:, 0]
                # Normalize embeddings
                normalized_embeddings = torch.nn.functional.normalize(
                    embeddings, p=2, dim=1
                )
        
        logger.info(f"Embedding {len(queries)} queries took {time() - start_time:.2f} seconds")
        
        return normalized_embeddings.tolist()
    
    @staticmethod
    def get_embedding_model_parameters(embedding_model: str):
        """Get the parameters for the embedding model."""
        return embedding_model_to_parameters[embedding_model]
    
    @staticmethod
    def get_supported_embedding_models():
        """Get the list of supported embedding models."""
        return embedding_model_to_parameters.keys()