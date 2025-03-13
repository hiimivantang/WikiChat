import argparse
import asyncio
import os
import pathlib
import re
import sys
import tarfile
import pickle
import time
import queue
import signal
import random
import gc  # For explicit garbage collection
from multiprocessing import Process, Queue, cpu_count
from urllib.parse import unquote

import orjson
from bs4 import BeautifulSoup, NavigableString
from markdownify import MarkdownConverter
from mwparserfromhtml.parse.plaintext import html_to_plaintext
import orjsonl
from tqdm import tqdm
import json
import pdb

from preprocessing.block import Block

sys.path.insert(0, "./")
from pipelines.utils import get_logger
from preprocessing.utils import (
    batch_get_wikidata_english_name,
    draw_and_save_histogram_log_bins,
    find_forest_roots_and_members,
    get_from_translation_map,
    load_translation_map,
    num_tokens,
    save_translation_map,
    translation_prefix,
)
from preprocessing.wikipedia_disambiguation import is_disambiguation

logger = get_logger(__name__)
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity(transformers_logging.ERROR)

# Constants for redirection map and processing
REDIRECTION_MAP_FILE = "redirection_map.pkl"
QUEUE_TIMEOUT = 300  # 5 minutes timeout for queue operations
WORKER_TIMEOUT = 600  # 10 minutes timeout for worker processes
BATCH_SIZE = 10000    # Number of blocks to save in each batch
DEFAULT_QUEUE_SIZE = 5000  # Default size for bounded queues
DEFAULT_WORKER_COUNT = 18  # Default number of worker processes for a high-core system

inverse_redirection_map = (
    {}
)  # used to expand translation search. This map includes a map of each root to itself, for simplicity
frequent_words_to_exclude = set()


def save_redirection_map(file_path, redirection_map):
    """Save the redirection map to a file using pickle."""
    temp_file = f"{file_path}.tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(redirection_map, f)
    os.replace(temp_file, file_path)
    logger.info(f"Redirection map saved to {file_path}")


def load_redirection_map(file_path):
    """Load the redirection map from a file using pickle."""
    if not os.path.exists(file_path):
        logger.info(f"No redirection map file found at {file_path}")
        return None
    
    try:
        with open(file_path, "rb") as f:
            redirection_map = pickle.load(f)
        logger.info(f"Loaded redirection map with {len(redirection_map)} entries")
        return redirection_map
    except Exception as e:
        logger.warning(f"Error loading redirection map: {str(e)}")
        return None


def build_inverse_redirection_map(redirection_map):
    """Build the inverse redirection map from the redirection map."""
    inverse_map = {}
    for node, redirectors in redirection_map.items():
        for node2 in redirectors:
            inverse_map[node2] = node
    logger.info(f"Built inverse redirection map with {len(inverse_map)} entries")
    return inverse_map


def batch_writer_process(batch_queue, output_path, batch_size):
    """
    Separate process for writing batches to disk.
    This allows the main process to continue processing without being blocked by I/O.
    
    Args:
        batch_queue: Queue containing batches of blocks to write
        output_path: Base path for output files
        batch_size: Maximum size of each batch
    """
    batch_num = 1
    current_batch = []
    
    logger.info(f"Batch writer process started, writing to {output_path}")
    
    try:
        while True:
            try:
                # Get batch item with timeout
                item = batch_queue.get(timeout=QUEUE_TIMEOUT)
                
                # None signals end of processing
                if item is None:
                    logger.info("Batch writer received termination signal")
                    break
                
                # Add to current batch
                current_batch.append(item)
                
                # Write batch if it reaches the target size
                if len(current_batch) >= batch_size:
                    batch_file = f"{output_path}.part{batch_num}"
                    logger.info(f"Writing batch {batch_num} with {len(current_batch)} blocks")
                    orjsonl.save(batch_file, current_batch)
                    current_batch = []
                    batch_num += 1
                    
            except queue.Empty:
                # Check if we have a partial batch to write during timeout
                if current_batch:
                    batch_file = f"{output_path}.part{batch_num}"
                    logger.info(f"Writing partial batch {batch_num} with {len(current_batch)} blocks after timeout")
                    orjsonl.save(batch_file, current_batch)
                    current_batch = []
                    batch_num += 1
                logger.debug("Batch writer timeout - waiting for more data")
                
    except Exception as e:
        logger.error(f"Error in batch writer process: {str(e)}")
    finally:
        # Write any remaining blocks
        if current_batch:
            try:
                batch_file = f"{output_path}.part{batch_num}"
                logger.info(f"Writing final batch {batch_num} with {len(current_batch)} blocks")
                orjsonl.save(batch_file, current_batch)
            except Exception as e:
                logger.error(f"Error writing final batch: {str(e)}")
        
        logger.info(f"Batch writer process completed after writing {batch_num} batches")


banned_sections = {
    "en": [
        "See also",
        "References",
        "External links",
        "Notes",
        "Sources",
        "Categories",
        "Further reading",
        "Citations",
        "Footnotes",
    ],
    "fa": [
        "همچنین ببینید",
        "پانویس",
        "منابع",
        "پیوند به بیرون",
        "یادداشت‌ها",
        "منابع و پانویس",
        "رده‌ها",
        "مطالعه بیشتر",
        "جستارهای وابسته",
    ],
    "es": [
        "Véase también",
        "Referencias",
        "Enlaces externos",
        "Notas",
        "Fuentes",
        "Categorías",
        "Lecturas adicionales",
        "Notas al pie",
    ],
    "fr": [
        "Voir aussi",
        "Références",
        "Liens externes",
        "Notes",
        "Sources",
        "Catégories",
        "Lecture complémentaire",
        "Notes et références",
    ],
    "it": [
        "Vedi anche",
        "Note",
        "Riferimenti",
        "Collegamenti esterni",
        "Fonti",
        "Categorie",
        "Bibliografia",
        "Altri progetti",
    ],
    "de": [
        "Siehe auch",
        "Einzelnachweise",
        "Weblinks",
        "Anmerkungen",
        "Quellen",
        "Kategorien",
        "Literatur",
        "Fußnoten",
    ],
    "ja": [
        "関連項目",
        "脚注",
        "注釈",
        "出典",
        "参考文献",
        "外部リンク",
        "参照",
        "参照リンク",
    ],
    "ru": [
        "См. также",
        "Примечания",
        "Ссылки",
        "Источники",
        "Литература",
        "Категории",
        "Дополнительные сведения",
        "Примечания",
    ],
    "pt": [
        "Ver também",
        "Referências",
        "Ligações externas",
        "Notas",
        "Fontes",
        "Categorias",
        "Leitura adicional",
        "Notas de rodapé",
    ],
    "zh": ["参见", "参考文献", "外部链接", "注释", "来源", "分类", "延伸阅读", "脚注"],
}
all_banned_sections = set(
    [s for language in banned_sections for s in banned_sections[language]]
)


def compress_markup(markdown_text: str):
    """
    Replaces multiple spaces and tabls with just one space. This does not affect how Markup is displayed
    """
    return re.sub(r"[ \t]+", " ", markdown_text)


def is_banned_section(title_stack: list[str]) -> bool:
    if len(title_stack) == 2 and title_stack[-1] in all_banned_sections:
        return True
    return False


def find_h_tags_hierarchy(tag):
    hierarchy = []
    current_level = float("inf")  # Start with an infinitely deep level
    for sibling in tag.find_all_previous(["h1", "h2", "h3", "h4", "h5", "h6"]):
        # Stop if another table is encountered
        level = int(sibling.name[1])  # Extract the numeric level of the header
        if level < current_level:
            hierarchy.append(sibling)
            current_level = level
    return hierarchy[::-1]  # Reverse to maintain the order from top to bottom


def tag_to_markdown(table, article_title: str) -> tuple[str, str]:
    md = MarkdownConverter().convert_soup(table)
    md = compress_markup(md)
    md = md.strip()
    hierarchy = [h.text for h in find_h_tags_hierarchy(table)]
    full_section_title = " > ".join([article_title] + hierarchy)

    return (full_section_title, md)


def find_table_descriptions(tag) -> tuple[str, str]:
    """
    Finds descriptions (<dl> tags) before and after a given table element within an HTML document.

    Args:
        tag: The BeautifulSoup tag object representing a table in an HTML document.

    Returns:
        A tuple of two strings: The first string contains the text of the description list
        found immediately before the table (if any), and the second string contains the text
        of the description list found immediately after the table (if any). If no description
        list is found in a respective position, an empty string is returned for that position.
    """
    pre_dl = ""
    post_dl = ""
    # Iterate through previous siblings of the tag
    for sibling in tag.previous_siblings:
        # Check if the sibling is a NavigableString and not empty or just whitespace
        if sibling.name in ["dl"] and sibling.text and sibling.text.strip():
            pre_dl = sibling.text.strip()
            break
    for sibling in tag.next_siblings:
        if (
            hasattr(sibling, "name")
            and sibling.name == "dl"
            and sibling.text
            and sibling.text.strip()
        ):
            post_dl = sibling.text.strip()
            break
    return pre_dl, post_dl


def get_tables_and_infoboxes(
    html_soup: BeautifulSoup, article_title: str, extra_tables: list
) -> list[Block]:

    blocks = []
    tables = set(html_soup.select("table.sidebar, table.wikitable") + extra_tables)
    infoboxes = html_soup.find_all(
        "table", class_=lambda x: (x and "infobox" in x)
    )  # french uses infobox_v2, which this pattern also matches

    for block_type, tag_list in zip(["table", "infobox"], [tables, infoboxes]):
        for tag in tag_list:
            try:
                full_section_title, content = tag_to_markdown(tag, article_title)
                if block_type == "table":
                    pretable, post_table = find_table_descriptions(tag)
                    if pretable:
                        content = pretable + "\n" + content
                    if post_table:
                        content = content + "\n" + post_table
                blocks.append(
                    Block(
                        content_string=content,
                        full_section_title=full_section_title,
                        block_type=block_type,
                    )
                )
            except Exception as e:
                logger.debug(
                    "BeautifulSoup encountered an error while parsing article '%s': %s",
                    article_title,
                    str(e),
                )
                continue

    return blocks


def get_passages(
    html_soup: BeautifulSoup,
    article_title: str,
    pack_to_tokens: int,
    exclude_elements={
        "Reference",
        "ExternalLink",
        "Heading",
        "Category",
        "Citation",
        "Media",
        "Navigation",
        "Note",
        "Messagebox",
        "Infobox",
        "Wikitable",
        "Comment",
        "Source",
        "Table",
    },
) -> list[tuple[str, str]]:
    """
    Extract plaintext from the HTML object in a depth-first manner,
    including full path to a section in headings.

    Args:
        article_title: The title of the article (or root section).
        exclude_elements: Set of HTML element types to exclude.

    Returns:
        A tuple of (heading, plaintext) where heading is the full path to the section.
    """
    section_stack = [
        article_title
    ]  # Initialize stack with the article title as the root section.
    blocks = []

    def get_full_heading():
        return " > ".join(section_stack)

    for i, section in enumerate(html_soup.findAll("section")):
        # Construct heading with full path
        if i != 0:  # Skip the first section since it's the article title itself
            current_heading = section.findChild().text
            if len(section_stack) == 1:
                section_stack.append(
                    current_heading
                )  # Direct subsection of the article
            else:
                if len(section_stack) < 1:
                    logger.warning(
                        "Section structure in article '%s' is malformed.", article_title
                    )
                else:
                    section_stack[-1] = (
                        current_heading  # Replace the last section title with the current one
                    )

        # get plaintext for each paragraph in the section
        plaintext = ""
        prev_para_context = "pre-first-para"
        for (
            node_plaintext,
            _,
            element_types,
            para_context,
        ) in html_to_plaintext(section):
            # Check for nested sections to update heading path
            if element_types.count("Section") > 1:
                if node_plaintext not in section_stack:
                    section_stack.append(
                        node_plaintext
                    )  # Nest deeper for new subsections
                else:
                    if len(section_stack) > 0:
                        section_stack.pop()  # Ascend as we exit a subsection
                    else:
                        logger.warning(
                            "Hierarchy of sections for article %s ran into an error.",
                            article_title,
                        )
                break

            if is_banned_section(section_stack) or (
                exclude_elements and exclude_elements.intersection(element_types)
            ):
                continue
            if node_plaintext == "\n" and set(element_types) == {"Section"}:
                if plaintext.strip():
                    blocks.append(
                        Block(
                            content_string=plaintext,
                            full_section_title=get_full_heading(),
                            block_type="text",
                        )
                    )
                plaintext = ""
                prev_para_context = para_context
            elif para_context != prev_para_context:
                if plaintext.strip():
                    blocks.append(
                        Block(
                            content_string=plaintext,
                            full_section_title=get_full_heading(),
                            block_type="text",
                        )
                    )

                plaintext = node_plaintext
                prev_para_context = para_context
            else:
                plaintext += node_plaintext
                prev_para_context = para_context

        if plaintext.strip():
            blocks.append(
                Block(
                    content_string=plaintext,
                    full_section_title=get_full_heading(),
                    block_type="text",
                )
            )

        # Reset or ascend the section stack as necessary
        if i != 0 and len(section_stack) > 2:
            section_stack.pop()  # Ascend when leaving a subsection to its parent

    blocks = pack_blocks(blocks, pack_to_tokens)

    return blocks


def pack_blocks(blocks: list[tuple[str, str]], pack_to_tokens: int) -> list[Block]:
    """
    Passages is the list of tuples where each tuple is (subsection title, passage).

    This function concatenates consecutive passages with the same subsection
    title as long as their combined length does not exceed `pack_to_tokens` tokens.
    """

    if not blocks:
        return []

    packed_blocks = []
    current_block = blocks[0]

    current_block.num_tokens = num_tokens(
        current_block.full_section_title + " " + current_block.content_string
    )
    for next_block in blocks[1:]:
        # Check if the next block has the exact same section title and does not exceed the character limit
        num_tokens_after_merge = current_block.num_tokens + num_tokens(
            "\n" + next_block.content_string
        )
        if (
            next_block.full_section_title == current_block.full_section_title
            and num_tokens_after_merge < pack_to_tokens
        ):
            current_block.content_string += (
                " " + next_block.content_string
            )  # Concatenate blocks with a space in between
            current_block.num_tokens = num_tokens_after_merge
        else:
            # Once a block reaches the limit or a new title is found, append the current state and move on
            packed_blocks.append(current_block)
            current_block = next_block
            current_block.num_tokens = num_tokens(
                current_block.full_section_title + " " + current_block.content_string
            )

    # Adding the last accumulated paragraph
    packed_blocks.append(current_block)

    return packed_blocks


def get_entity_translation_to_english(
    source_language: str, entity_name: str, context: str = ""
) -> str:
    """
    The output of this function can be safely used to replace entity_name
    Args:
        source_language: The language code of the source entity name.
        entity_name: The name of the entity to translate.
        context: Optional; a string within which the presence of the translated name
                 is checked to avoid redundancy. Defaults to an empty string.

    Returns:
        A string containing the original entity name and its English translation,
        separated by a specific prefix, if the translation is found and deemed
        non-redundant. Returns just the entity name if the translation is redundant or not found.
    """
    cached_english = get_from_translation_map(
        source_language, entity_name, inverse_redirection_map
    )
    if cached_english is not None:
        if cached_english not in frequent_words_to_exclude:
            # remove parenthesis in entities like `XYZ (singer)`
            parenthesis_index = cached_english.find("(")
            if parenthesis_index >= 0:
                cached_english = cached_english[:parenthesis_index].strip()
            if (
                len(cached_english) > 0
                and cached_english.lower()
                not in context.lower()  # don't add hint if the `context` already contains it
            ):

                return f"{entity_name} {translation_prefix}{cached_english})"
            else:
                return entity_name
        else:
            logger.debug("Excluded '%s' because it is too frequent", cached_english)
            return entity_name
    else:
        logger.debug(
            "Did not find link entity in Wikidata for %s",
            entity_name,
        )
        return entity_name


def preprocess_links(
    html_soup: BeautifulSoup, article_title: str, should_translate: bool, language: str
) -> None:
    for a_tag in html_soup.find_all("a", href=True):
        if a_tag["href"].endswith("&action=edit"):
            # delete Wikipedia links like "edit this article" etc.
            a_tag.decompose()
            continue
        if should_translate and a_tag["href"].startswith("./"):
            # internal link to a Wikipedia article
            entity_name = url_to_entity_name(a_tag["href"])
            a_tag.replace_with(
                NavigableString(
                    get_entity_translation_to_english(
                        language, entity_name, context=article_title
                    )
                )
            )
        else:
            # external link, or internal link that doesn't need translation
            a_tag.replace_with(NavigableString(a_tag.text))


def get_adjacent_tags(
    soup, tag_that_comes_first: str, tag_that_comes_second: str
) -> tuple:
    tags_coming_first = []
    tags_coming_second = []

    for tag_1 in soup.find_all(tag_that_comes_first):
        next_sibling = tag_1.find_next_sibling()

        if next_sibling and next_sibling.name == tag_that_comes_second:
            tags_coming_second.append(next_sibling)
            tags_coming_first.append(tag_1)

    return tags_coming_first, tags_coming_second


def prepend_dls(html_soup):
    """
    In Wikipedia, often <dl> tags are used incorrectly instead of <p> or even <h3>.
    This function is a heuristic and imperfect way of connecting <dl> tags to their relevant context.
    """
    filtered_dl_tags = set()
    dls, posts = get_adjacent_tags(html_soup, "dl", "ul")
    for dl, post in zip(dls, posts):
        # Check if the <dl> tag is not a descendant of a <table> tag
        if not dl.find_parent("table") and not dl.find_all("table"):
            # Ensure none of the descendants have a class "mwe-math-element"
            if not dl.find_all(class_="mwe-math-element"):
                filtered_dl_tags.add(dl)
                post.insert(0, NavigableString(dl.text + "\n"))

    dls, posts = get_adjacent_tags(html_soup, "dl", "p")
    for dl, post in zip(dls, posts):
        # Check if the <dl> tag is not a descendant of a <table> tag
        if not dl.find_parent("table") and not dl.find_all("table"):
            # Ensure none of the descendants have a class "mwe-math-element"
            if not dl.find_all(class_="mwe-math-element"):
                filtered_dl_tags.add(dl)
                post.insert(0, NavigableString(dl.text + "\n"))

    for tag in filtered_dl_tags:
        tag.decompose()


def process_articles(
    input_queue,
    output_queue,
    dead_letter_queue,
    pack_to_tokens: int,
    language: str,
    should_translate: bool,
    worker_id: int,
):
    """Process articles from the input queue with improved error handling and timeouts."""
    # Set up a timeout handler
    def timeout_handler(signum, frame):
        logger.warning(f"Worker {worker_id} timed out while processing an article")
        # We'll just continue to the next article
        return
    
    signal.signal(signal.SIGALRM, timeout_handler)
    
    while True:
        try:
            # Use a timeout when getting from the queue to avoid indefinite blocking
            try:
                article = input_queue.get(timeout=QUEUE_TIMEOUT)
                if article is None:
                    logger.info(f"Worker {worker_id} received termination signal")
                    break
            except queue.Empty:
                logger.warning(f"Worker {worker_id} timed out waiting for input")
                continue
                
            # Set an alarm for processing this article
            signal.alarm(WORKER_TIMEOUT)
            
            article_blocks = []
            try:
                html = article["article_body"]["html"]
                article_title = article["name"]

                if should_translate:
                    # add English translation to title
                    article_title = get_entity_translation_to_english(
                        language, article_title, context=article_title
                    )  # don't add the translation if the article_title already has or is in English

                html_soup = BeautifulSoup(html, features="lxml")

                # Remove all citations and style tags
                for tag in html_soup.select("sup.reference, style"):
                    tag.decompose()

                # Display math equations better
                for tag in html_soup.find_all(
                    "math", alttext=lambda value: value and value.startswith("{\displaystyle")
                ):
                    tag.replace_with(
                        NavigableString(tag["alttext"][len("{\displaystyle") : -1])
                    )
                preprocess_links(
                    html_soup,
                    article_title,
                    should_translate=should_translate,
                    language=language,
                )

                # <dl> right after <table>
                tables1, dls1 = get_adjacent_tags(
                    html_soup, tag_that_comes_first="table", tag_that_comes_second="dl"
                )
                # <table> right after <dl>
                dls2, tables2 = get_adjacent_tags(
                    html_soup, tag_that_comes_first="dl", tag_that_comes_second="table"
                )
                article_blocks.extend(
                    get_tables_and_infoboxes(html_soup, article_title, tables1 + tables2)
                )

                # sidebars are already processed together with tables
                # https://en.wikipedia.org/wiki/Template:Sidebar
                # https://en.wikipedia.org/wiki/Template:Clade
                for t in html_soup.select(
                    "table.sidebar, table.clade, figure, .shortdescription"
                ):
                    t.decompose()
                # <dl> tags before or after tables are already indcluded with the table, so remove them here
                for dl in dls1 + dls2:
                    dl.decompose()

                prepend_dls(html_soup)

                article_blocks.extend(
                    get_passages(
                        html_soup=html_soup,
                        article_title=article_title,
                        pack_to_tokens=pack_to_tokens,
                    )
                )
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing article: {str(e)}")
                try:
                    dead_letter_queue.put(article, timeout=QUEUE_TIMEOUT)
                except queue.Full:
                    logger.error(f"Worker {worker_id} couldn't add to dead letter queue: queue full")
                continue

            for block in article_blocks:
                if len(block.content_string) < 3:
                    continue  # skip empty blocks

                if block.num_tokens == 0:
                    block.num_tokens = num_tokens(
                        block.full_section_title + " " + block.content_string
                    )
                block.article_title = article_title
                block.language = language
                block.last_edit_date = article["date_modified"]
                if should_translate:
                    block.deduplicate_translations()

                # Use a timeout for putting into the output queue
                try:
                    output_queue.put(block, timeout=QUEUE_TIMEOUT)
                except queue.Full:
                    logger.warning(f"Worker {worker_id} timed out putting block in output queue: queue full")
                    # If we can't put in the output queue, we'll add to dead letter queue
                    try:
                        dead_letter_queue.put(article, timeout=QUEUE_TIMEOUT)
                    except queue.Full:
                        logger.error(f"Worker {worker_id} couldn't add to dead letter queue either: queue full")
            
            # Cancel the alarm since we finished processing this article
            signal.alarm(0)
            
        except KeyError as e:
            logger.warning(f"Worker {worker_id} encountered KeyError: {str(e)}")
            try:
                dead_letter_queue.put(article, timeout=QUEUE_TIMEOUT)
            except queue.Full:
                logger.error(f"Worker {worker_id} couldn't add to dead letter queue after KeyError: queue full")
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered unexpected error: {str(e)}")
            try:
                dead_letter_queue.put(article, timeout=QUEUE_TIMEOUT)
            except queue.Full:
                logger.error(f"Worker {worker_id} couldn't add to dead letter queue after error: queue full")
    
    # Signal completion
    try:
        output_queue.put(None, timeout=QUEUE_TIMEOUT)  # signal the end
        logger.info(f"Worker {worker_id} finished and sent termination signal")
    except queue.Full:
        logger.error(f"Worker {worker_id} couldn't send termination signal: queue full")


def url_to_entity_name(url):
    ret = unquote(url).split("/")[-1].replace("_", " ")
    if ret.endswith("?action=edit&redlink=1"):
        ret = ret[: -len("?action=edit&redlink=1")]
    return ret


def build_redirection_map(file_path: str) -> dict:
    redirection_incoming_edges: dict[str, set[str]] = (
        {}
    )  # maps an article url with all urls that redirect to it, via one or multiple hops

    for article in tqdm(
        tarfile_loader(file_path),
        desc="Building the Wikipedia redirection graph",
        miniters=1e-6,
        unit_scale=1,
        unit=" Articles",
        smoothing=0,
    ):
        if is_disambiguation(article):
            continue

        url = url_to_entity_name(article["url"])
        if url not in redirection_incoming_edges:
            redirection_incoming_edges[url] = set()
        if "redirects" in article:
            # Add redirects even if we have already seen it. This way, multi-hop redirects will be handled correctly.
            for redirect in article["redirects"]:
                redirected_url = url_to_entity_name(redirect["url"])
                redirection_incoming_edges[url].add(redirected_url)

    # print("before multihop consolidation: ", len(redirection_incoming_edges))
    # the structure of this dictionary describes a forest (i.e. collection of trees), with each item describing the incoming edges of a node
    # we want to find the root of all trees
    redirect_map = find_forest_roots_and_members(redirection_incoming_edges)
    # print("after multihop consolidation: ", len(redirect_map))
    return redirect_map


def tarfile_loader(file_path: str):
    """
    Generator that sequentially loads articles from a tar.gz file containing NDJSON formatted articles.
    Skips over articles with identifiers that have been seen already, ignoring redirects or duplicates.
    """
    tar_file_ = None
    try:
        tar_file_ = tarfile.open(file_path, mode="r|gz")
        while True:
            try:
                ndjson_file = tar_file_.next()
                if ndjson_file is None:
                    break
                else:
                    with tar_file_.extractfile(ndjson_file) as file_content:
                        for line in file_content:
                            try:
                                article = orjson.loads(line)
                                yield article
                            except Exception as e:
                                logger.warning(f"Error parsing article JSON: {str(e)}")
                                continue
            except Exception as e:
                logger.warning(f"Error processing tar file entry: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Fatal error in tarfile_loader: {str(e)}")
        raise
    finally:
        if tar_file_:
            try:
                tar_file_.close()
            except:
                pass


def estimate_article_complexity(article):
    """
    Estimate the processing complexity of an article to improve work distribution.
    Returns a score that can be used to prioritize and balance work.
    
    Args:
        article: The article dictionary
        
    Returns:
        A numerical score indicating estimated processing complexity
    """
    try:
        # Get HTML content length as proxy for processing complexity
        html_length = len(article["article_body"]["html"])
        
        # Add weight for articles with many redirects
        redirect_count = len(article.get("redirects", []))
        
        # Add weight for articles with tables or infoboxes that require more processing
        # Look for common table indicators in the HTML
        table_indicator_count = article["article_body"]["html"].count("<table")
        
        # Calculate complexity score - these weights could be tuned with profiling
        complexity = (
            html_length * 0.8 +  # Base HTML size is main factor
            redirect_count * 2000 +  # Each redirect adds complexity
            table_indicator_count * 5000  # Tables add significant processing time
        )
        
        return complexity
    except Exception:
        # Return a default medium complexity if estimation fails
        return 500000


def articles_without_disambiguation_or_redirections(
    file_path: str,
    num_workers: int,
    queue,
    redirect_map: dict,
    max_articles: int,
):
    """Feed articles to workers with improved error handling and work balancing."""
    # the reason we iterate over and process the Wikipedia dump file again is we don't want to keep everything in memory, especially for large dump files.
    pbar = None
    try:
        pbar = tqdm(
            desc="Extracting blocks",
            miniters=1e-6,
            unit_scale=1,
            unit=" Blocks",
            smoothing=0,
            total=len(redirect_map) if max_articles < 0 else min(max_articles, len(redirect_map)),
        )
        counter = 0
        
        # Buffer articles for better distribution - storing (article, complexity) tuples
        article_buffer = []
        buffer_size = min(num_workers * 3, 100)  # Buffer enough to distribute well, but not too much memory
        
        for article in tarfile_loader(file_path):
            try:
                if is_disambiguation(article):
                    continue
                url = url_to_entity_name(article["url"])
                if url not in redirect_map:
                    continue
                
                # Estimate article complexity for better work distribution
                complexity = estimate_article_complexity(article)
                
                # Add to buffer
                article_buffer.append((article, complexity))
                
                # When buffer is full, sort by complexity and distribute to workers
                if len(article_buffer) >= buffer_size:
                    # Sort by complexity (alternating small and large for better balance)
                    article_buffer.sort(key=lambda x: x[1])
                    distributed_buffer = []
                    
                    # Interleave small and large articles for better distribution
                    # This helps prevent all large articles from going to the same worker
                    for i in range(len(article_buffer) // 2):
                        distributed_buffer.append(article_buffer[i])  # Small article
                        large_idx = len(article_buffer) - 1 - i
                        if large_idx > i:  # Avoid duplicates if odd length
                            distributed_buffer.append(article_buffer[large_idx])  # Large article
                    
                    # Add any remaining article if buffer length is odd
                    if len(article_buffer) % 2 == 1:
                        distributed_buffer.append(article_buffer[len(article_buffer) // 2])
                    
                    # Now put articles in queue in distributed order
                    for article_item, _ in distributed_buffer:
                        try:
                            # Use timeout to avoid indefinite blocking
                            queue.put(article_item, timeout=QUEUE_TIMEOUT)
                            pbar.update(1)
                            counter += 1
                            if max_articles > 0 and counter >= max_articles:
                                break
                        except queue.Full:
                            logger.warning("Timeout putting article in queue: queue full")
                            # If we can't add to the queue, we'll slow down a bit and retry
                            time.sleep(1)
                            try:
                                queue.put(article_item, timeout=QUEUE_TIMEOUT)
                                pbar.update(1)
                                counter += 1
                            except queue.Full:
                                logger.error("Failed to add article to queue even after retry")
                    
                    # Clear buffer after processing
                    article_buffer = []
                    
                    # Exit early if we've hit max_articles
                    if max_articles > 0 and counter >= max_articles:
                        break
                        
            except Exception as e:
                logger.warning(f"Error processing article: {str(e)}")
                continue
        
        # Process any remaining articles in buffer
        for article_item, _ in article_buffer:
            try:
                queue.put(article_item, timeout=QUEUE_TIMEOUT)
                pbar.update(1)
                counter += 1
                if max_articles > 0 and counter >= max_articles:
                    break
            except queue.Full:
                logger.warning("Queue full when processing remaining articles")
                time.sleep(1)  # Brief pause to let queue drain
                try:
                    queue.put(article_item, timeout=QUEUE_TIMEOUT)
                    pbar.update(1)
                    counter += 1
                except queue.Full:
                    logger.error("Failed to add remaining article to queue even after retry")
                
    except Exception as e:
        logger.error(f"Error in article extraction: {str(e)}")
    finally:
        # Ensure we always send termination signals to workers
        logger.info("Sending termination signals to workers")
        for _ in range(num_workers):
            try:
                queue.put(None, timeout=QUEUE_TIMEOUT)  # signal end to all workers
            except queue.Full:
                logger.error("Failed to send termination signal to a worker")
        
        # Always close the progress bar
        if pbar:
            pbar.close()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="A Wikipedia HTML dump, which is a tar.gz file containing multiple .ndjson files",
    )
    arg_parser.add_argument("--output_path", type=str, required=True)
    arg_parser.add_argument("--language", type=str, required=True)
    arg_parser.add_argument(
        "--should_translate",
        action="store_true",
        help="If we should translate named entities to English using Wikidata. Has no effect if `--language` is English",
    )
    arg_parser.add_argument(
        "--wikidata_translation_map",
        type=str,
        help="Where to read/write the translation mapping we obtain from Wikidata.",
    )
    arg_parser.add_argument("--num_workers", type=int, default=DEFAULT_WORKER_COUNT, 
                        help=f"Number of worker processes. Default is {DEFAULT_WORKER_COUNT}, optimized for high-core systems.")
    arg_parser.add_argument(
        "--pack_to_tokens",
        type=int,
        default=0,
        help="If consecutive paragraphs in the same subsection are small, we greedily concatenate them together, while keeping the result shorter than this many tokens."
        " This helps reduce the number of vector embeddings when indexing. BAAI/bge-m3 tokenizer is used to determine token boundaries.",
    )
    arg_parser.add_argument(
        "--max_articles",
        type=int,
        default=-1,
        help="Will stop after processing this many articles. -1 means no limit. Used for testing.",
    )
    arg_parser.add_argument(
        "--num_exclude_frequent_words_from_translation",
        type=int,
        default=0,
        help="Will exclude translations for the top N most frequent words used in the English Wikipedia.",
    )
    arg_parser.add_argument(
        "--queue_size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help=f"Size of the processing queues. Larger values improve throughput but use more memory. Default: {DEFAULT_QUEUE_SIZE}",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of blocks to include in each batch file. Default: {BATCH_SIZE}",
    )
    arg_parser.add_argument(
        "--parallel_batch_writing",
        action="store_true",
        help="Write batches in a separate process to avoid blocking the main processing loop.",
    )

    args = arg_parser.parse_args()
    if args.language == "en":
        args.should_translate = False
        
    # Define the redirection map file path, using the same path and naming convention as v2
    redirection_map_file = os.path.join("checkpoints", f"{args.language}_redirection_map.pkl")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Try to load redirection map or build it if needed
    redirection_map = None
    if os.path.exists(redirection_map_file):
        logger.info(f"Loading redirection map from {redirection_map_file}")
        redirection_map = load_redirection_map(redirection_map_file)
    
    if redirection_map is None:
        logger.info("Building redirection map")
        redirection_map = build_redirection_map(args.input_path)
        save_redirection_map(redirection_map_file, redirection_map)
    
    # Build inverse redirection map
    inverse_redirection_map = build_inverse_redirection_map(redirection_map)

    if args.should_translate:
        logger.info(
            "Number of articles, excluding disambiguation and redirection pages: %d",
            len(redirection_map),
        )
        if args.num_exclude_frequent_words_from_translation > 0:
            try:
                with open("preprocessing/word_list.txt") as f:
                    for line in f:
                        frequent_words_to_exclude.add(line.strip())
                        if (
                            len(frequent_words_to_exclude)
                            >= args.num_exclude_frequent_words_from_translation
                        ):
                            break
            except Exception as e:
                logger.error(f"Error loading frequent words list: {str(e)}")

        load_translation_map(args.wikidata_translation_map)
        non_cached_titles = []
        for url in redirection_map:
            if (
                get_from_translation_map(args.language, url, inverse_redirection_map)
                is None
            ):
                non_cached_titles.append(url)

        if len(non_cached_titles) > 0:
            logger.info(
                "Did not find %d articles in the translation map, will call the Wikidata API for them",
                len(non_cached_titles),
            )
            asyncio.run(
                batch_get_wikidata_english_name(non_cached_titles, args.language)
            )
            save_translation_map(args.wikidata_translation_map)

    # Use bounded queues with configurable sizes
    logger.info(f"Using queue size of {args.queue_size} and batch size of {args.batch_size}")
    input_queue = Queue(maxsize=args.queue_size)
    output_queue = Queue(maxsize=args.queue_size)
    dead_letter_queue = Queue(maxsize=args.queue_size)
    
    # Set up batch queue if using parallel batch writing
    batch_queue = None
    if args.parallel_batch_writing:
        batch_queue = Queue(maxsize=args.queue_size)
        logger.info("Using parallel batch writing")
    
    all_worker_processes = []

    # make parent directories
    pathlib.Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)

    # Create worker processes with worker_id
    for worker_id in range(args.num_workers):
        all_worker_processes.append(
            Process(
                target=process_articles,
                args=(
                    input_queue,
                    output_queue,
                    dead_letter_queue,
                    args.pack_to_tokens,
                    args.language,
                    args.should_translate,
                    worker_id,  # Pass worker_id
                ),
            )
        )

    # The process that feeds the articles to workers
    reader_process = Process(
        target=articles_without_disambiguation_or_redirections,
        args=(
            args.input_path,
            args.num_workers,
            input_queue,
            redirection_map,
            args.max_articles,
        ),
    )
    
    # Create batch writer process if parallel batch writing is enabled
    batch_writer = None
    if args.parallel_batch_writing:
        batch_writer = Process(
            target=batch_writer_process,
            args=(
                batch_queue,
                args.output_path,
                args.batch_size,
            ),
        )

    # Variables for batch writing and statistics
    workers_finished = 0
    text_count, table_count, infobox_count = 0, 0, 0
    current_batch = []
    batch_num = 1
    # Don't store all blocks in memory - just track statistics
    total_blocks = 0
    total_tokens = 0
    token_histogram_data = []  # Store sampling of tokens for histogram
    token_sample_rate = 0.01  # Only store 1% of token counts for histogram
    counter = 0
    start_time = time.time()
    last_progress_time = start_time
    last_gc_time = start_time
    last_log_time = start_time
    last_counter = 0  # For accurate blocks/second calculation
    
    # Constants for GC and logging
    GC_INTERVAL = 60  # Run garbage collection every 60 seconds
    LOG_INTERVAL = 30  # Log progress every 30 seconds
    DETAILED_LOG_INTERVAL = 300  # Detailed stats every 5 minutes

    try:
        # Start all processes
        processes_to_start = all_worker_processes + [reader_process]
        if batch_writer:
            processes_to_start.append(batch_writer)
            
        for p in processes_to_start:
            p.start()

        # Main processing loop with better error handling
        while workers_finished < len(all_worker_processes):
            try:
                # Periodic logging with reduced frequency
                current_time = time.time()
                
                # Basic progress logging at regular intervals
                if current_time - last_log_time > LOG_INTERVAL:
                    # Calculate blocks/second based on blocks processed since last log
                    recent_blocks = counter - last_counter
                    elapsed_time = current_time - last_log_time
                    recent_blocks_per_second = recent_blocks / max(0.1, elapsed_time)
                    
                    # Calculate overall statistics
                    overall_blocks_per_second = counter / max(0.1, current_time - start_time)
                    
                    logger.info(f"Progress: {counter:,d} blocks, rate: {recent_blocks_per_second:.1f} blocks/s " +
                               f"(avg: {overall_blocks_per_second:.1f}), " +
                               f"workers: {(len(all_worker_processes) - workers_finished)}/{len(all_worker_processes)}")
                    
                    # Update last counter for next calculation
                    last_counter = counter
                    last_log_time = current_time
                
                # Detailed statistics less frequently to reduce overhead
                if current_time - last_progress_time > DETAILED_LOG_INTERVAL:
                    # Only log detailed stats occasionally
                    logger.info(f"Detailed stats: text={text_count:,d}, table={table_count:,d}, " +
                               f"infobox={infobox_count:,d}, avg tokens={total_tokens/max(1, total_blocks):.1f}")
                    last_progress_time = current_time
                
                # Periodic garbage collection to prevent memory buildup
                if current_time - last_gc_time > GC_INTERVAL:
                    logger.debug("Running garbage collection")
                    gc.collect()
                    last_gc_time = current_time
                
                try:
                    # Get block with timeout
                    block = output_queue.get(timeout=QUEUE_TIMEOUT)
                    
                    if block is None:
                        workers_finished += 1
                        logger.info(f"Worker completed. {workers_finished}/{len(all_worker_processes)} workers finished")
                        continue
                        
                    # Process the block
                    if block.block_type == "text":
                        text_count += 1
                    elif block.block_type == "table":
                        table_count += 1
                    elif block.block_type == "infobox":
                        infobox_count += 1
                    else:
                        logger.warning(f"Unknown block type: {block.block_type}")
                        continue
                        
                    # Convert to JSON but don't store all blocks in memory
                    json_block = block.to_json(counter)
                    current_batch.append(json_block)  # Add to current batch only
                    
                    # Update statistics instead of storing the block
                    if block.block_type == "text":
                        text_count += 1
                    elif block.block_type == "table":
                        table_count += 1
                    elif block.block_type == "infobox":
                        infobox_count += 1
                        
                    # Track token stats for histogram (sample only a percentage to save memory)
                    total_tokens += block.num_tokens
                    total_blocks += 1
                    if random.random() < token_sample_rate:  # Only store a sample of token counts
                        token_histogram_data.append(block.num_tokens)
                        
                    counter += 1
                    
                    # Save batch if it reaches the batch size
                    if len(current_batch) >= args.batch_size:
                        if args.parallel_batch_writing:
                            # Send batch to parallel writer process
                            for block in current_batch:
                                try:
                                    batch_queue.put(block, timeout=QUEUE_TIMEOUT)
                                except queue.Full:
                                    logger.warning("Batch queue full, waiting for space")
                                    time.sleep(0.5)  # Brief pause to let queue drain
                                    batch_queue.put(block, timeout=QUEUE_TIMEOUT)
                        else:
                            # Write batch directly
                            batch_file = f"{args.output_path}.part{batch_num}"
                            logger.info(f"Saving batch {batch_num} with {len(current_batch)} blocks")
                            orjsonl.save(batch_file, current_batch)
                            batch_num += 1
                            
                            # Force garbage collection after writing a batch
                            # This helps prevent memory fragmentation
                            if batch_num % 5 == 0:  # Only collect every 5 batches to balance performance
                                gc.collect()
                            
                        current_batch = []
                        
                except queue.Empty:
                    logger.warning("Timeout waiting for blocks from workers")
                    # Check if all workers are still alive
                    alive_workers = sum(1 for p in all_worker_processes if p.is_alive())
                    if alive_workers == 0 and workers_finished < len(all_worker_processes):
                        logger.warning("All workers appear to be dead but not all reported completion")
                        break
            except Exception as e:
                logger.error(f"Error in main processing loop: {str(e)}")
                        
        # Save any remaining blocks in the current batch
        if current_batch:
            if args.parallel_batch_writing:
                # Send remaining blocks to batch writer
                for block in current_batch:
                    try:
                        batch_queue.put(block, timeout=QUEUE_TIMEOUT)
                    except queue.Full:
                        logger.warning("Batch queue full when finalizing, waiting for space")
                        time.sleep(0.5)
                        batch_queue.put(block, timeout=QUEUE_TIMEOUT)
            else:
                # Write directly
                batch_file = f"{args.output_path}.part{batch_num}"
                logger.info(f"Saving final batch {batch_num} with {len(current_batch)} blocks")
                orjsonl.save(batch_file, current_batch)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main process: {str(e)}")
    finally:
        # Send termination signal to batch writer if using parallel batch writing
        if args.parallel_batch_writing and batch_queue:
            logger.info("Sending termination signal to batch writer")
            try:
                batch_queue.put(None, timeout=QUEUE_TIMEOUT)
            except queue.Full:
                logger.error("Failed to send termination signal to batch writer")
        
        # Ensure processes are terminated properly
        logger.info("Cleaning up processes")
        processes_to_cleanup = all_worker_processes + [reader_process]
        if batch_writer:
            processes_to_cleanup.append(batch_writer)
            
        for p in processes_to_cleanup:
            if p.is_alive():
                p.terminate()
                logger.info(f"Terminated process {p.pid}")
        
        # Wait for processes to complete with timeout
        for p in processes_to_cleanup:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Process {p.pid} did not terminate gracefully")
                
        # We don't have all blocks in memory anymore, so we'll have to concatenate the part files
        logger.info(f"Processing complete with {total_blocks} total blocks")
        logger.info(f"The individual part files contain all the processed blocks")
        logger.info(f"If you want a single combined file, use: cat {args.output_path}.part* > {args.output_path}")

    # Process dead letter queue
    dlq_path = os.path.join(os.path.dirname(args.output_path), "dead_letter_queue.jsonl")
    logger.info(f"Processing dead letter queue to {dlq_path}")
    
    try:
        with open(dlq_path, "w") as f:
            dlq_count = 0
            # Use timeout when getting from queue
            while True:
                try:
                    dlq_article = dead_letter_queue.get(timeout=10)
                    if dlq_article is None:
                        break
                    else:
                        f.write(json.dumps(dlq_article) + "\n")
                        dlq_count += 1
                except queue.Empty:
                    logger.info("No more items in dead letter queue or timeout reached")
                    break
                    
        logger.info(f"Saved {dlq_count} items to dead letter queue")
    except Exception as e:
        logger.error(f"Error processing dead letter queue: {str(e)}")
    
    # Save the collection size
    try:
        size_file = os.path.join(os.path.dirname(args.output_path), "collection_size.txt")
        with open(size_file, "w") as f:
            f.write(str(total_blocks))
        logger.info(f"Saved collection size ({total_blocks}) to {size_file}")
    except Exception as e:
        logger.error(f"Error saving collection size: {str(e)}")

    # Log statistics
    logger.info("Found {:,d} text blocks (including lists)".format(text_count))
    logger.info("Found {:,d} table blocks".format(table_count))
    logger.info("Found {:,d} infobox blocks".format(infobox_count))
    logger.info(
        "Total number of blocks: {:,d}".format(text_count + table_count + infobox_count)
    )

    # Generate histogram using sampled token data
    try:
        histogram_path = args.output_path.rsplit(".", 1)[0] + "_histogram.png"
        if token_histogram_data:
            logger.info(f"Generating histogram from {len(token_histogram_data)} sampled blocks (of {total_blocks} total)")
            draw_and_save_histogram_log_bins(
                token_histogram_data,
                histogram_path
            )
            logger.info(f"Saved histogram to {histogram_path}")
        else:
            logger.warning("No token data available for histogram generation")
    except Exception as e:
        logger.error(f"Error generating histogram: {str(e)}")
        
    logger.info(f"Processing completed with {total_blocks:,d} blocks, average {total_tokens/max(1, total_blocks):.1f} tokens per block")
