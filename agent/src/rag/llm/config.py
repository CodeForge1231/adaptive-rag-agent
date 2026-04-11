from dataclasses import dataclass


@dataclass(frozen=True)
class BaseLLMConfig:
    """
    Base configuration for Large Language Model providers.
    Defines common parameters shared across various LLM implementations.
    """
    provider: str
    """The unique identifier of the LLM provider (e.g., 'openai', 'ollama', 'anthropic')."""

    temperature: float
    """
    Sampling temperature that controls the degree of randomness in the output.
    Lower values result in more deterministic responses, while higher values increase creativity.
    """


@dataclass(frozen=True)
class OpenAILLMConfig(BaseLLMConfig):
    """
    Configuration specific to OpenAI's hosted language models.
    """
    model: str
    """The specific OpenAI model version to be used (e.g., 'gpt-4o', 'gpt-3.5-turbo')."""


@dataclass(frozen=True)
class OllamaLLMConfig(BaseLLMConfig):
    """
    Configuration for models served through a local or remote Ollama instance.
    """
    model: str
    """The model name available within the Ollama environment (e.g., 'llama3', 'mistral')."""

    host: str = "localhost"
    """The network hostname or IP address where the Ollama service is running."""

    port: int = 11434
    """The communication port for the Ollama API server."""

    ssl: bool = False
    """Enables or disables SSL/TLS encryption for the connection to the Ollama server."""