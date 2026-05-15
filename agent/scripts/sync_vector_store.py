import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Configure logging for the ingestion process
logger = logging.getLogger(__name__)

# Add project root to sys.path for local module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import bootstrap

def load_json_data(file_name: str) -> list[dict]:
    """
    Load a JSON dataset from the local filesystem.

    Parameters
    ----------
    file_name : str
        The name or relative path of the JSON file to be loaded.

    Returns
    -------
    list of dict
        The parsed contents of the JSON file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist in the expected directory.
    """
    # Resolve the absolute path to the data file
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / file_name
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {file_path}")

    logger.info(f"Loading JSON dataset from {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_documents(data: list[dict], use_chunks: bool, chunk_size: int, chunk_overlap: int):
    """
    Convert raw dictionary data into LangChain Document objects.

    Parameters
    ----------
    data : list of dict
        The raw records loaded from the JSON source.
    use_chunks : bool
        Whether to split documents into smaller fragments.
    chunk_size : int
        Maximum number of characters per document chunk.
    chunk_overlap : int
        Number of overlapping characters between adjacent chunks.

    Returns
    -------
    tuple
        A pair containing (list of Document objects, list of UUID strings).
    """
    logger.info("Preparing documents from JSON dataset")

    documents = []
    ids = []

    # Initialize the text splitter if chunking is enabled
    splitter = None
    if use_chunks:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    for item in data:
        text = item.get("text", "")
        meta = item.get("metadata", {})

        doc = Document(page_content=text, metadata=meta)

        if use_chunks:
            # Generate overlapping chunks for better retrieval coverage
            chunks = splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{uuid.uuid4()}")
        else:
            # Keep the document in its original size
            documents.append(doc)
            ids.append(str(uuid.uuid4()))

    logger.info(f"Prepared {len(documents)} documents")
    return documents, ids


async def main(json_path: str, use_chunks: bool, chunk_size: int, chunk_overlap: int):
    """
    Orchestrate the ingestion pipeline from source file to vector store.

    Parameters
    ----------
    json_path : str
        Path to the source JSON dataset.
    use_chunks : bool
        Flag to enable text splitting.
    chunk_size : int
        The size of text segments.
    chunk_overlap : int
        The overlap between segments.
    """
    # Boot application context and services
    app_context = await bootstrap()
    vectorstore = app_context.vectorstore
    
    # Load and parse the source dataset
    data = load_json_data(json_path)

    # Transform records into document objects for the vector database
    documents, ids = prepare_documents(
        data, use_chunks=use_chunks, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Clear existing data and perform batch upload
    await vectorstore.aclear_all_documents()
    await vectorstore.aadd_documents(documents, ids=ids)

    logger.info("JSON ingestion pipeline completed")


if __name__ == "__main__":
    # Define and parse command line interface arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--use-chunks", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)

    args = parser.parse_args()

    # Run the ingestion pipeline
    asyncio.run(
        main(
            json_path=args.json_path,
            use_chunks=args.use_chunks,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    )