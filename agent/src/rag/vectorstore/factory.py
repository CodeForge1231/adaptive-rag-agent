from .chroma import ChromaVectorStore
from .config import ChromaVectorStoreConfig


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    _config_map = {"chroma": ChromaVectorStoreConfig}

    @classmethod
    def create(cls, raw_cfg: dict, embeddings):
        """
        Create a vector store instance from a raw configuration dictionary.

        Args:
            raw_cfg (dict): Raw configuration data containing provider and settings.
            embeddings: An object providing access to the embedding model instance.

        Returns:
            BaseVectorStore: A configured instance of a vector store implementation.

        Raises:
            ValueError: If the specified provider is not supported.
            RuntimeError: If the configuration cannot be mapped to a creation strategy.
        """
        provider = raw_cfg.get("provider")
        # Resolve vector store  config
        try:
            cfg_cls = cls._config_map[provider]
        except KeyError:
            raise ValueError(f"Unknown vector store provider: {provider}")

        # Parse and validate config
        cfg = cfg_cls(**raw_cfg)

        # Instantiate vector store based on config
        match cfg:
            case ChromaVectorStoreConfig(host=str()):
                return ChromaVectorStore.create_server(
                    embeddings=embeddings.get_model(),
                    cfg=cfg,
                )

            case ChromaVectorStoreConfig(persist_path=str()):
                return ChromaVectorStore.create_local(
                    embeddings=embeddings.get_model(),
                    cfg=cfg,
                )

            case _:
                raise RuntimeError(f"Unhandled vectorstore config: {type(cfg)}")
