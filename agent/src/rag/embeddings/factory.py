from .base import BaseEmbeddingModel
from .config import (
    HuggingFaceEmbeddingConfig,
    OllamaEmbeddingConfig,
    OpenAIEmbeddingConfig,
)
from .hugging_face import HuggingFaceModel
from .ollama import OllamaModel
from .openai import OpenAIModel


class EmbeddingFactory:
    """
    Factory responsible for creating embedding model instances
    based on validated configuration.
    """

    # Mapping from provider name to config class
    _config_map = {
        "openai": OpenAIEmbeddingConfig,
        "ollama": OllamaEmbeddingConfig,
        "huggingface": HuggingFaceEmbeddingConfig,
    }

    @classmethod
    def create(cls, raw_cfg: dict) -> BaseEmbeddingModel:
        """Create an embedding model from raw configuration."""
        provider = raw_cfg.get("provider")

        # Resolve embedding config
        try:
            cfg_cls = cls._config_map[provider]
        except KeyError:
            raise ValueError(f"Unknown embedding provider: {provider}")

        # Parse config
        cfg = cfg_cls(**raw_cfg)

        # Instantiate embedding model based on config type
        match cfg:
            case OpenAIEmbeddingConfig():
                return OpenAIModel(
                    model_name=cfg.model,
                )

            case OllamaEmbeddingConfig():
                return OllamaModel(
                    model_name=cfg.model,
                    host=cfg.host,
                    port=cfg.port,
                    ssl=cfg.ssl,
                )

            case HuggingFaceEmbeddingConfig():
                return HuggingFaceModel(
                    model_name=cfg.model,
                    device=cfg.device,
                )
            case _:
                raise RuntimeError(f"Unhandled embedding config: {type(cfg)}")
