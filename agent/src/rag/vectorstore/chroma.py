import asyncio
import logging

import chromadb
import numpy as np
from langchain_chroma import Chroma

from src.rag.schemas.vectorstore import VectorStoreData

from .base import BaseVectorStore
from .config import ChromaVectorStoreConfig

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Chroma implementation of the vector store interface."""

    def __init__(self, db_instance: Chroma):
        # Underlying LangChain Chroma instance
        self._db = db_instance

    async def aadd_documents(self, documents, ids=None):
        """
        Asynchronously add documents to the Chroma collection.

        Args:
            documents (List): List of document objects to be embedded and stored.
            ids (Optional[List[str]]): List of unique identifiers for the documents.

        Raises:
            Exception: If the document insertion fails.
        """
        count = len(documents)
        logger.info(f"Adding {count} documents to the vector store...")

        try:
            await self._db.aadd_documents(documents=documents, ids=ids)
            logger.info(f"Successfully added {count} documents to the vector store.")
        except Exception as e:
            logger.error(f"Failed to add documents to the vector store: {e}")
            raise e

    def get_all_data(self) -> VectorStoreData:
        """
        Fetch all available data (embeddings, texts, metadata) from the collection.

        Returns:
            VectorStoreData: A dictionary containing IDs, embeddings, documents, and metadatas.
        """
        # Retrieve raw data using the internal collection object
        raw = self._db._collection.get(include=["embeddings", "documents", "metadatas"])
        return {
            "ids": raw["ids"],
            "embeddings": np.array(raw["embeddings"]),
            "documents": raw["documents"],
            "metadatas": raw["metadatas"],
        }

    @classmethod
    def create_local(cls, embeddings, cfg: ChromaVectorStoreConfig):
        """
        Factory method to initialize a persistent local Chroma instance.

        Args:
            embeddings: The embedding model instance.
            cfg (ChromaVectorStoreConfig): Configuration containing path and collection name.

        Returns:
            ChromaVectorStore: An instance of this class configured for local storage.
        """
        db = Chroma(
            persist_directory=cfg.persist_path,
            collection_name=cfg.collection,
            embedding_function=embeddings,
        )
        return cls(db)

    @classmethod
    def create_server(cls, embeddings, cfg: ChromaVectorStoreConfig):
        """
        Factory method to initialize a Chroma instance connected to a remote server.

        Args:
            embeddings: The embedding model instance.
            cfg (ChromaVectorStoreConfig): Configuration containing host, port, and security settings.

        Returns:
            ChromaVectorStore: An instance of this class configured for remote client access.
        """
        client = chromadb.HttpClient(
            host=cfg.host,
            port=cfg.port,
            ssl=cfg.ssl,
        )
        db = Chroma(
            client=client,
            collection_name=cfg.collection,
            embedding_function=embeddings,
        )
        return cls(db)

    async def asimilarity_search(self, query, *, k, filters=None):
        """
        Execute an asynchronous similarity search.

        Args:
            query (str): The search input string.
            k (int): Number of nearest neighbors to retrieve.
            filters (Optional[Dict]): Metadata filters to restrict search results.

        Returns:
            List: A list of relevant documents.
        """
        return await self._db.asimilarity_search(query, k=k, filter=filters)

    async def ammr_search(self, query, *, k, fetch_k, filters=None):
        """
        Execute an asynchronous Maximal Marginal Relevance (MMR) search.

        Args:
            query (str): The search input string.
            k (int): Number of final documents to return.
            fetch_k (int): Number of initial documents to fetch before re-ranking.
            filters (Optional[Dict]): Metadata filters to apply.

        Returns:
            List: A list of diverse and relevant documents.
        """
        return await self._db.amax_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, filter=filters
        )

    async def adelete_collection(self) -> None:
        """
        Asynchronously delete the entire collection from the database.
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._db.delete_collection)
            logger.info("Successfully deleted the entire collection.")
        except Exception as e:
            logger.error(f"Error during async collection deletion: {e}")

    async def aclear_all_documents(self) -> None:
        """
        Asynchronously remove all documents from the collection without deleting the index.
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: self._db._collection.get(include=[]))
            ids = results.get("ids", [])

            if ids:
                # Delete documents by IDs
                await loop.run_in_executor(None, lambda: self._db._collection.delete(ids=ids))
                logger.info(f"Cleared {len(ids)} documents from the collection.")
            else:
                logger.info("Collection is already empty. Nothing to clear.")
        except Exception as e:
            logger.error(f"Error while clearing documents: {e}")
