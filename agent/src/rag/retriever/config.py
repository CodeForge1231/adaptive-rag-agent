from dataclasses import dataclass


@dataclass(frozen=True)
class BaseRetrieverConfig:
    """
    Abstract base configuration for document retrieval components.
    """

    provider: str
    """The identifier for the retrieval strategy (e.g., 'simple', 'mmr')."""

    top_k: int
    """The total number of documents to return to the LLM context after processing."""


@dataclass(frozen=True)
class SimpleRetrieverConfig(BaseRetrieverConfig):
    """
    Configuration for standard similarity-based retrieval.
    """

    pass


@dataclass(frozen=True)
class MMRRetrieverConfig(BaseRetrieverConfig):
    """
    Configuration for Maximal Marginal Relevance (MMR) retrieval.
    """

    fetch_k: int
    """
    The initial number of candidate documents to retrieve from the vector store 
    prior to performing the diversity re-ranking process.
    """
