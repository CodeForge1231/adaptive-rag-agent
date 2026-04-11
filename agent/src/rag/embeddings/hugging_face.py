from langchain_huggingface import HuggingFaceEmbeddings

from .base import BaseEmbeddingModel


class HuggingFaceModel(BaseEmbeddingModel):
    """
    HuggingFace-based embedding model wrapper.
    Uses sentence-transformers or compatible HF models.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"
    ):
        """
        Initialize the Hugging Face embedding model.

        Parameters
        ----------
        model_name : str, default "sentence-transformers/all-MiniLM-L6-v2"
            The model identifier from the Hugging Face Hub or a local path.
        device : str, default "cpu"
            The hardware device to use for inference (e.g., 'cpu', 'cuda', 'mps').
        """
        model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
        super().__init__(model)

    async def aembed_documents(self, texts):
        """
        Embed multiple documents asynchronously.

        Parameters
        ----------
        texts : list[str]
            A list of strings to be converted into embedding vectors.

        Returns
        -------
        list[list[float]]
            A list of document embeddings.
        """
        return await self._model.aembed_documents(texts)

    async def aembed_query(self, text):
        """
        Embed a single query asynchronously.

        Parameters
        ----------
        text : str
            The query string to be converted into an embedding vector.

        Returns
        -------
        list[float]
            The query embedding vector.
        """
        return await self._model.aembed_query(text)
