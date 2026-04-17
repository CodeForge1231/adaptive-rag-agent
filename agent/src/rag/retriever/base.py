from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseRetriever(ABC):
    """
    Abstract base class for document retrieval services.
    """
    def __init__(self, vector_store, filter_adapter):
        """
        Initialize the retriever with its core dependencies.

        Parameters
        ----------
        vector_store : Any
            The underlying storage engine for vector embeddings.
        filter_adapter : BaseFilterAdapter
            The adapter used to handle metadata-based filtering logic.
        """
        self.vector_store = vector_store
        self.filter_adapter = filter_adapter

    @abstractmethod
    async def aretrieve(self, query: str, *, filters: Optional[Dict] = None):
        """
        Asynchronously search and retrieve relevant documents.

        Parameters
        ----------
        query : str
            The input string or search query used to perform similarity search.
        filters : Optional[Dict], default=None
            Generic metadata filters to refine the search results.

        Returns
        -------
        List[Document]
            A collection of retrieved documents matching the criteria.
        """
        pass
