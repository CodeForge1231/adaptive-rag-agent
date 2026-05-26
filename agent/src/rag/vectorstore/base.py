from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.rag.schemas.vectorstore import VectorStoreData


class BaseVectorStore(ABC):
    """
    Abstract Base Class for vector store implementations.
    """

    @abstractmethod
    async def aadd_documents(self, documents: List, ids: List): 
        """
        Asynchronously add documents and their corresponding IDs to the vector store.

        Args:
            documents (List): A list of document objects or content to be embedded.
            ids (List): Unique identifiers for each document.
        """
        pass

    @abstractmethod
    def get_all_data(self) -> VectorStoreData:
        """
        Retrieve all stored data from the vector store collection.

        Returns:
            VectorStoreData: An object containing the collection's data.
        """
        pass

    @abstractmethod
    async def asimilarity_search(
        self, query: str, *, k: int, filters: Optional[Dict] = None
    ) -> List:
        """
        Perform an asynchronous similarity search based on a query string.

        Args:
            query (str): The search query.
            k (int): Number of top-k similar documents to return.
            filters (Optional[Dict]): Metadata filters to apply to the search.

        Returns:
            List: A list of documents most similar to the query.
        """
        pass

    @abstractmethod
    async def ammr_search(
        self, query: str, *, k: int, fetch_k: int, filters: Optional[Dict] = None
    ) -> List:
        """
        Perform an asynchronous Maximal Marginal Relevance (MMR) search.
        
        MMR balances the trade-off between relevance to the query and 
        diversity among the results.

        Args:
            query (str): The search query.
            k (int): Number of final documents to return.
            fetch_k (int): Number of initial documents to fetch for re-ranking.
            filters (Optional[Dict]): Metadata filters to apply.

        Returns:
            List: A list of diverse yet relevant documents.
        """
        pass

    @abstractmethod
    def adelete_collection(self) -> None:
        """
        Asynchronously delete the entire collection or index from the vector store.
        """
        pass

    @abstractmethod
    def aclear_all_documents(self) -> None:
        """
        Asynchronously remove all documents from the current collection 
        without deleting the collection itself.
        """
        pass
