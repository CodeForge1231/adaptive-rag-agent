from .base import BaseRetriever
from .config import (
    MMRRetrieverConfig,
    SimpleRetrieverConfig,
)
from .filters.chroma import ChromaFilterAdapter
from .mmr import MMRRetriever
from .simple import SimpleRetriever


class RetrieverFactory:
    """
    Factory responsible for initializing document retrievers based on configuration.

    Attributes
    ----------
    _config_map : dict
        Mapping of retriever strategy names to their respective configuration dataclasses.
    _filter_adapter_map : dict
        Mapping of vector store provider names to their filter transformation logic.
    """

    # Mapping from provider name to config class
    _config_map = {
        "simple": SimpleRetrieverConfig,
        "mmr": MMRRetrieverConfig,
    }

    # Mapping from vector store provider to filter adapter
    _filter_adapter_map = {
        "chroma": ChromaFilterAdapter,
    }

    @classmethod
    def create(cls, raw_cfg: dict, vector_store, vectorstore_provider: str) -> BaseRetriever:
        """
        Create a concrete retriever instance from a raw configuration dictionary.
        
        Parameters
        ----------
        raw_cfg : dict
            The raw configuration parameters for the retriever.
        vector_store : Any
            The underlying vector database instance.
        vectorstore_provider : str
            The name of the vector store backend (e.g., "chroma").

        Returns
        -------
        BaseRetriever
            An initialized instance of a class implementing the BaseRetriever interface.

        Raises
        ------
        ValueError
            If the retriever provider specified in the config is unknown.
        RuntimeError
            If a valid configuration is provided but no matching implementation is handled.
        """
        provider = raw_cfg.get("provider")

        # Resolve retriever config
        try:
            cfg_cls = cls._config_map[provider]
        except KeyError:
            raise ValueError(f"Unknown retriever provider: {provider}")

        # Parse and validate config
        cfg = cfg_cls(**raw_cfg)

        # Optional vector store–specific filter adapter
        adapter_cls = cls._filter_adapter_map.get(vectorstore_provider)
        filter_adapter = adapter_cls() if adapter_cls else None

        # Instantiate retriever based on config type
        match cfg:
            case SimpleRetrieverConfig():
                return SimpleRetriever(
                    vector_store=vector_store,
                    filter_adapter=filter_adapter,
                    top_k=cfg.top_k,
                )

            case MMRRetrieverConfig():
                return MMRRetriever(
                    vector_store=vector_store,
                    filter_adapter=filter_adapter,
                    top_k=cfg.top_k,
                    fetch_k=cfg.fetch_k,
                )
            case _:
                raise RuntimeError(f"Unhandled retriever config: {type(cfg)}")
