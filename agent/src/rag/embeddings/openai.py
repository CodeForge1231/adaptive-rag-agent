from langchain_openai import OpenAIEmbeddings

from .base import BaseEmbeddingModel


class OpenAIModel(BaseEmbeddingModel):
    """
    OpenAI-based embedding model wrapper.
    """
    def __init__(self, model_name: str = "text-embedding-3-small"):
        model = OpenAIEmbeddings(model=model_name)
        super().__init__(model)

    async def aembed_documents(self, texts):
        """
        Embed multiple documents asynchronously.

        Parameters
        ----------
        texts : list[str]
            A list of strings to be converted into high-dimensional vectors.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors corresponding to the input texts.
        """
        return await self._model.aembed_documents(texts)

    async def aembed_query(self, text):
        """
        Embed a single query asynchronously.

        Parameters
        ----------
        text : str
            The query string to be converted into a vector representation.

        Returns
        -------
        list[float]
            The query embedding vector.
        """
        return await self._model.aembed_query(text)
