from dataclasses import dataclass
from pathlib import Path


# database
@dataclass(frozen=True)
class BaseDatabaseConfig:
    """
    Abstract base configuration for database connectivity.
    """
    provider: str
    """The database dialect or provider name (e.g., 'sqlite', 'postgres')."""

@dataclass(frozen=True)
class SQLiteDatabaseConfig(BaseDatabaseConfig):
    """
    Configuration for local SQLite database instances.
    """
    path: Path
    """The filesystem directory path where the SQLite database file is stored."""
    
    database: str
    """The filename of the database (e.g., 'app_data' without extension)."""


@dataclass(frozen=True)
class PostgresDatabaseConfig(BaseDatabaseConfig):
    """
    Configuration for remote PostgreSQL database instances.
    """
    host: str
    """The network hostname or IP address of the PostgreSQL server."""
    
    port: int
    """The TCP port number on which the database server is listening."""
    
    user: str
    """The username used for database authentication."""
    
    password: str
    """The sensitive password associated with the database user."""
    
    database: str
    """The name of the specific database schema to connect to."""
    
    ssl: bool
    """Enables or disables SSL/TLS encryption for the database connection."""


# encryption
@dataclass(frozen=True)
class BaseEncryptionConfig:
    """
    Abstract base configuration for data encryption services.
    """
    provider: str
    """The identifier for the encryption algorithm or library (e.g., 'fernet')."""


@dataclass(frozen=True)
class FernetEncryptionConfig(BaseEncryptionConfig):
    """
    Configuration for Fernet symmetric encryption.
    """
    key: str
    """The base64-encoded secret key required for cryptographic operations."""


# document history
@dataclass(frozen=True)
class DocumentHistoryConfig:
    """
    Configuration for managing document interaction history.
    """
    history_size: int
    """The maximum number of document interaction records retained per user session."""