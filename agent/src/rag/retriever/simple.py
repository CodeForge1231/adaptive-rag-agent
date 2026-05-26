from typing import Dict, Optional

from .base import BaseRetriever


class SimpleRetriever(BaseRetriever):
    """
    Retriever implementation for basic similarity-based search.

    Attributes
    ----------
    top_k : int
        The maximum number of documents to retrieve.
    """

    def __init__(self, vector_store, filter_adapter, top_k, **kwargs):
        """
        Initialize the simple retriever with search constraints.

        Parameters
        ----------
        vector_store : Any
            The vector database instance supporting asynchronous similarity search.
        filter_adapter : BaseFilterAdapter, optional
            The adapter for translating generic filters to vector store syntax.
        top_k : int
            The number of results to return.
        """
        super().__init__(vector_store, filter_adapter)
        self.top_k = top_k

    async def aretrieve(self, query: str, *, filters: Optional[Dict] = None):
        """
        Asynchronously retrieve documents based on vector similarity.

        The method handles the transformation of filters if both the filters 
        and an adapter are provided, then executes the search against 
        the vector store.

        Parameters
        ----------
        query : str
            The user query or embedding search string.
        filters : Optional[Dict], default=None
            Metadata filters to narrow the search scope.

        Returns
        -------
        List[Document]
            A list of the most similar documents retrieved from the store.
        """
        # Resolve and transform filters if applicable
        adapted_filters = (
            self.filter_adapter.transform(filters) if filters and self.filter_adapter else None
        )

        # Execute standard similarity search on the vector database
        return await self.vector_store.asimilarity_search(
            query, k=self.top_k, filters=adapted_filters
        )
