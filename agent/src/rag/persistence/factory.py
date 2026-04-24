import logging

from src.rag.persistence.config import (
    DocumentHistoryConfig,
    FernetEncryptionConfig,
    PostgresDatabaseConfig,
    SQLiteDatabaseConfig,
)
from src.rag.persistence.encryption.fernet import FernetEncryptionService
from src.rag.persistence.sqlalchemy.base import Base
from src.rag.persistence.sqlalchemy.engine import create_session
from src.rag.persistence.sqlalchemy.repositories import (
    SQLAlchemyDocumentHistoryRepository,
    SQLAlchemyUserProfileRepository,
)

from .base import PersistenceContext

logger = logging.getLogger(__name__)


class PersistenceFactory:
    """
    Factory responsible for initializing the complete persistence layer.
    """

    # Mapping from database provider to config class
    _db_config_map = {
        "sqlite": SQLiteDatabaseConfig,
        "postgresql": PostgresDatabaseConfig,
    }

    # Mapping from encryption provider to implementation
    _encryption_map = {
        "fernet": FernetEncryptionService,
    }

    @classmethod
    async def create(cls, raw_cfg: dict) -> PersistenceContext:
        """
        Initialize the database, encryption, and repositories from a raw configuration.
        
        Parameters
        ----------
        raw_cfg : dict
            A dictionary containing nested configurations for 'database', 
            'encryption', and 'documents'.

        Returns
        -------
        PersistenceContext
            A container holding the active database engine and initialized 
            repository instances.

        Raises
        ------
        ValueError
            If an unsupported database or encryption provider is specified.
        """
        db_raw = raw_cfg["database"]
        provider = db_raw["provider"]

        try:
            db_cfg_cls = cls._db_config_map[provider]
        except KeyError:
            raise ValueError(f"Unsupported database provider: {provider}")

        db_cfg = db_cfg_cls(**db_raw)

        # Create async engine and session factory
        engine, session_factory = create_session(db_cfg)

        # Create database schema
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Encryption configuration
        enc_raw = raw_cfg["encryption"]
        enc_provider = enc_raw["provider"]

        try:
            enc_cls = cls._encryption_map[enc_provider]
        except KeyError:
            raise ValueError(f"Unsupported encryption provider: {enc_provider}")

        enc_cfg = FernetEncryptionConfig(**enc_raw)
        encryption = enc_cls(enc_cfg.key)

        # Document history configuration
        doc_cfg = DocumentHistoryConfig(**raw_cfg["documents"])

        # Initialize repositories
        repositories = {
            "user_profiles": SQLAlchemyUserProfileRepository(
                session_factory,
                encryption,
            ),
            "document_history": SQLAlchemyDocumentHistoryRepository(
                doc_cfg,
                session_factory,
            ),
        }

        logger.info("Persistence initialized")

        return PersistenceContext(
            engine=engine,
            repositories=repositories,
        )
