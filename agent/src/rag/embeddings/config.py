from dataclasses import dataclass


@dataclass(frozen=True)
class BaseEmbeddingConfig:
    """
    Base configuration for embedding model providers.
    """
    provider: str
    """The identifier for the embedding service provider (e.g., 'openai', 'ollama', 'huggingface')."""

@dataclass(frozen=True)
class OpenAIEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for OpenAI's cloud-based embedding models.
    """
    model: str
    """The specific model ID to be used for generating vectors (e.g., 'text-embedding-3-small')."""


@dataclass(frozen=True)
class OllamaEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for embedding models served through a local or remote Ollama instance.
    """
    model: str
    """The name of the local model used for inference (e.g., 'nomic-embed-text')."""

    host: str = "localhost"
    """The network address where the Ollama API server is reachable."""

    port: int = 11434
    """The TCP port number dedicated to the Ollama service."""

    ssl: bool = False
    """Enables SSL/TLS encryption for secure communication with the Ollama server."""


@dataclass(frozen=True)
class HuggingFaceEmbeddingConfig(BaseEmbeddingConfig):
    """
    Configuration for local Transformer-based embedding models from Hugging Face.
    """
    model: str
    """The Hugging Face Hub repository ID or the absolute path to a local model directory."""

    device: str = "cpu"
    """ The computational hardware used for inference"""