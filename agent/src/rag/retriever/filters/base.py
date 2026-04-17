from abc import ABC, abstractmethod


class BaseFilterAdapter(ABC):
    """
    Abstract interface for vector store metadata filter adapters.

    This class defines the contract for transforming high-level, generic filter 
    dictionaries into the specific syntax or domain-specific language (DSL) 
    required by various vector database backends (e.g., Chroma, Pinecone, PGVector).

    Methods
    -------
    transform(filters)
        Converts generic filter logic into a backend-compatible format.
    """

    @abstractmethod
    def transform(self, filters: dict):
        """
        Transform generic filter expressions into a backend-specific structure.

        Parameters
        ----------
        filters : dict
            A dictionary representing the filter criteria (e.g., field names 
            and values or complex logical operators).

        Returns
        -------
        Any
            The transformed filter object ready for use in a vector store query.
        """
        pass
