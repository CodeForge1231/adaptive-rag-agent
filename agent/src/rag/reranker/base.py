from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseReranker(ABC):
    """
    Abstract base class for document reranking implementations.

    This class defines the interface for reranking services that refine 
    the relevance of a list of retrieved documents relative to a specific 
    query.

    Methods
    -------
    arerank(query, documents, k)
        Asynchronously reranks documents and returns the top k results.
    """

    @abstractmethod
    async def arerank(self, query: str, documents: List[Document], k: int) -> List[Document]:
        """
        Asynchronously rerank a list of documents based on query relevance.

        Parameters
        ----------
        query : str
            The user's search query or transformation of the query.
        documents : List[Document]
            A collection of LangChain Document objects to be reranked.
        k : int
            The number of top-scoring documents to return.

        Returns
        -------
        List[Document]
            A list containing the top k documents, ordered by relevance.
        """
        pass
