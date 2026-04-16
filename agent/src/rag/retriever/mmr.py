from typing import Dict, Optional

from src.core.observability import observability

from .base import BaseRetriever


class MMRRetriever(BaseRetriever):
    """
    Retriever using Maximal Marginal Relevance (MMR)
    to balance relevance and diversity.
    """

    def __init__(self, vector_store, filter_adapter, top_k, fetch_k, **kwargs):
        """
        Initialize the MMR retriever with search constraints.

        Parameters
        ----------
        vector_store : Any
            The vector database instance supporting asynchronous MMR search.
        filter_adapter : BaseFilterAdapter
            The adapter for translating generic filters to vector store syntax.
        top_k : int
            The number of results to return after diversity filtering.
        fetch_k : int
            The initial pool size for the candidate search.
        """
        super().__init__(vector_store, filter_adapter)
        self.top_k = top_k
        self.fetch_k = fetch_k

    @observability.time_it_async("Retrieve MMR")
    async def aretrieve(self, query: str, *, filters: Optional[Dict] = None):
        """
        Asynchronously perform a diversity-aware search.

        This method transforms high-level filters using the assigned adapter 
        and invokes the vector store's MMR implementation.

        Parameters
        ----------
        query : str
            The search query.
        filters : Optional[Dict], default=None
            Metadata filters to apply to the search.

        Returns
        -------
        List[Document]
            A list of reranked documents balancing relevance and diversity.
        """
        adapted_filters = self.filter_adapter.transform(filters)

        return await self.vector_store.ammr_search(
            query, k=self.top_k, fetch_k=self.fetch_k, filters=adapted_filters
        )
