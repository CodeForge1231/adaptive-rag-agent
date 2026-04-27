from dataclasses import dataclass


@dataclass(frozen=True)
class BaseRerankerConfig:
    """
    Abstract base configuration for document reranking services.
    """

    provider: str
    """The identifier for the reranking implementation """


@dataclass(frozen=True)
class LLMRerankerConfig(BaseRerankerConfig):
    """
    Configuration for rerankers that utilize Large Language Models.
    """
    
    top_k: int
    """
    The maximum number of high-relevance documents to retain after the LLM 
    has completed the scoring and re-ordering process.
    """
