from .base import BaseReranker
from .config import LLMRerankerConfig
from .llm_reranker import LLMReranker


class RerankerFactory:
    """
    Factory responsible for creating and configuring reranker instances.

    Attributes
    ----------
    _config_map : dict
        Internal mapping of provider identifiers to their corresponding 
        configuration dataclasses.
    """

    # Mapping from provider name to config class
    _config_map = {
        "llm": LLMRerankerConfig,
    }

    @classmethod
    def create(cls, raw_cfg: dict, chains_library) -> BaseReranker:
        """
        Create a reranker instance based on the provided configuration dictionary.

        This method validates the provider, initializes the appropriate 
        configuration object, and matches it against supported implementations.

        Parameters
        ----------
        raw_cfg : dict
            A dictionary containing reranker parameters, including the 'provider'.
        chains_library : Any
            A library or container providing access to the necessary 
            LangChain logic for reranking.

        Returns
        -------
        BaseReranker
            An initialized instance of a class implementing the BaseReranker interface.

        Raises
        ------
        ValueError
            If the specified provider is not found in the configuration map.
        RuntimeError
            If a valid configuration is provided but no matching 
            implementation is defined.
        """
        provider = raw_cfg.get("provider")

        # Resolve reranker config
        try:
            cfg_cls = cls._config_map[provider]
        except KeyError:
            raise ValueError(f"Unknown reranker provider: {provider}")

        # Parse and validate config
        cfg = cfg_cls(**raw_cfg)

        # Instantiate reranker based on config type
        match cfg:
            case LLMRerankerConfig():
                return LLMReranker(
                    rerank_chain=chains_library.reranker,
                    top_k=cfg.top_k,
                )
            case _:
                raise RuntimeError(f"Unhandled reranker config: {type(cfg)}")
