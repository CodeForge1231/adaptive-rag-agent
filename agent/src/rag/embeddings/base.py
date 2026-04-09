from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for embedding models.

    Defines a unified asynchronous interface for generating vector 
    representations of documents and queries across different providers.

    Attributes
    ----------
    _model : Any
        The underlying LangChain-compatible embedding model instance.
    """

    def __init__(self, model):
        # Underlying langchain-compatible embedding model
        self._model = model

    @abstractmethod
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents asynchronously.

        Parameters
        ----------
        texts : list[str]
            A list of document strings to be converted into vectors.

        Returns
        -------
        list[list[float]]
            A list of high-dimensional vector representations.
        """
        pass

    @abstractmethod
    async def aembed_query(self, text: str) -> list[float]:
        """
        Generate an embedding for a single query asynchronously.

        Parameters
        ----------
        text : str
            The query string to be converted into a vector.

        Returns
        -------
        list[float]
            A high-dimensional vector representing the query.
        """
        pass

    def get_model(self):
        """
        Return the underlying embedding model instance.

        Returns
        -------
        Any
            The raw embedding model object.
        """
        return self._model
