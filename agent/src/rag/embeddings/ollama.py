import logging

import ollama
from langchain_ollama import OllamaEmbeddings

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


class OllamaModel(BaseEmbeddingModel):
    """
    Ollama embedding model wrapper.
    Automatically pulls the model if it is not available locally.
    """

    def __init__(
        self,
        model_name: str = "llama3",
        host: str = "localhost",
        port: int = 11434,
        ssl: bool = False,
    ):
        """
        Initialize the Ollama embedding model and ensure availability.

        Parameters
        ----------
        model_name : str, default "llama3"
            The name of the model to use for embeddings.
        host : str, default "localhost"
            The hostname of the Ollama server.
        port : int, default 11434
            The port number for the Ollama API.
        ssl : bool, default False
            Whether to use HTTPS for the connection.

        Raises
        ------
        ollama.ResponseError
            If there is an unrecoverable error during model check or pull.
        """
        protocol = "https" if ssl else "http"
        base_url = f"{protocol}://{host}:{port}"
        client = ollama.Client(host=base_url)

        try:
            client.show(model_name)
        except ollama.ResponseError as e:
            logger.warning(f"Model '{model_name}' not found on Ollama server.")
            logger.info(f"Downloading (pulling) model '{model_name}'... This may take a while.")
            if e.status_code == 404:
                client.pull(model_name)
                logger.info(f"Model '{model_name}' pulled successfully.")
            else:
                logger.error(f"Failed to communicate with Ollama: {e}")
                raise e

        model = OllamaEmbeddings(model=model_name, base_url=base_url)
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
            A list of high-dimensional vectors representing the documents.
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
