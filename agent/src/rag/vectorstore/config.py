from dataclasses import dataclass


@dataclass(frozen=True)
class BaseVectorStoreConfig:
    """
    Base configuration for vector store providers.
    
    Defines common parameters required by any vector database implementation.
    """
    
    provider: str
    """The name of the vector database provider"""
    
    collection: str
    """The specific collection or index name within the vector store."""


@dataclass(frozen=True)
class ChromaVectorStoreConfig(BaseVectorStoreConfig):
    """
    Configuration specific to the Chroma vector store.
    """
    
    persist_path: str | None = None
    """The local file system directory where the database will be stored."""
    host: str | None = None
    """The hostname or IP address of the remote Chroma server."""

    port: int | None = None
    """The port number for the remote Chroma server connection."""
    
    ssl: bool = False
    """Flag to enable or disable SSL/TLS encryption for server communication."""