from .base import BaseLLM
from .config import OllamaLLMConfig, OpenAILLMConfig
from .ollama import OllamaModel
from .openai import OpenAIModel


class LLMFactory:
    """
    Factory responsible for creating LLM instances based on validated configuration.
    """

    # Mapping from provider name to config class
    _config_map = {
        "openai": OpenAILLMConfig,
        "ollama": OllamaLLMConfig,
    }

    @classmethod
    def create(cls, raw_cfg: dict) -> BaseLLM:
        """
        Create an LLM instance from raw configuration.

        Parameters
        ----------
        raw_cfg : dict
            A dictionary containing the necessary parameters for the LLM, 
            including 'provider', 'model', and 'temperature'.

        Returns
        -------
        BaseLLM
            An initialized instance of a class that implements the BaseLLM interface.

        Raises
        ------
        ValueError
            If the provided 'provider' is not supported in the config map.
        TypeError
            If the raw configuration contains invalid or missing arguments 
            for the target config class.
        """
        provider = raw_cfg.get("provider")

        # Resolve LLM config
        try:
            cfg_cls = cls._config_map[provider]
        except KeyError:
            raise ValueError(f"Unknown LLM provider: {provider}")

        # Parse config
        cfg = cfg_cls(**raw_cfg)

        # Instantiate model based on config type
        match cfg:
            case OpenAILLMConfig():
                return OpenAIModel(
                    model_name=cfg.model,
                    temperature=cfg.temperature,
                )

            case OllamaLLMConfig():
                return OllamaModel(
                    model_name=cfg.model,
                    host=cfg.host,
                    port=cfg.port,
                    ssl=cfg.ssl,
                    temperature=cfg.temperature,
                )
