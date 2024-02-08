import glob
import os
from langchain.document_loaders import BSHTMLLoader
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.html import partition_html

from rag_experiment_accelerator.doc_loader.structuredLoader import (
    load_structured_files,
)
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


def load_html_files(
    folder_path: str,
    chunk_size: str,
    overlap_size: str,
    glob_patterns: list[str] = ["html", "htm", "xhtml", "html5"],
):
    """
    Load and process HTML files from a given folder path.

    Args:
        folder_path (str): The path of the folder where files are located.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (list[str]): List of file extensions to consider (e.g., ["html", "htm", ...]).

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.debug("Loading html files")

    return load_structured_files(
        file_format="HTML",
        language="html",
        loader=BSHTMLLoader,
        folder_path=folder_path,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        glob_patterns=glob_patterns,
        loader_kwargs={"open_encoding": "utf-8"},
    )

def load_html_files_semantic(
    folder_path: str,
    chunk_size: str,
    overlap_size: str,
    glob_patterns: list[str] = ["html", "htm", "xhtml", "html5"],
):
    """
    Load and process HTML files from a given folder path.

    Args:
        folder_path (str): The path of the folder where files are located.
        chunk_size (str): The size of the chunks to split the documents into.
        overlap_size (str): The size of the overlapping parts between chunks.
        glob_patterns (list[str]): List of file extensions to consider (e.g., ["html", "htm", ...]).

    Returns:
        list[Document]: A list of processed and split document chunks.
    """

    logger.info(f"Loading html files from {folder_path}")
    matching_files = []
    for pattern in glob_patterns:
        # "." is used for hidden files, "~" is used for Word temporary files
        glob_pattern = f"**/[!.~]*.{pattern}"
        full_glob_pattern = os.path.join(folder_path, glob_pattern)
        matching_files += glob.glob(full_glob_pattern, recursive=True)

    logger.debug(f"Found {len(matching_files)} html files")

    documents = []
    if loader_kwargs is None:
        loader_kwargs = {}

    all_chunks = {}
    id = 0
    for file in matching_files:
        elements = partition_html(filename=file)
        chunks = chunk_elements(elements)
        for chunk in chunks:
            all_chunks[id] = chunk.text
            id += 1

    return all_chunks

