import json
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects import postgresql, sqlite

from src.rag.persistence.config import DocumentHistoryConfig
from src.rag.persistence.sqlalchemy.models import DocumentORM, UserProfileORM

from .base import DocumentHistoryRepository, UserProfileRepository


class SQLAlchemyUserProfileRepository(UserProfileRepository):
    """
    SQLAlchemy-based repository for persistent storage of encrypted user profiles.
    """

    def __init__(self, session_factory, encryption):
        """
        Initialize the repository with database and encryption dependencies.

        Parameters
        ----------
        session_factory : async_sessionmaker
            The session factory bound to the target database.
        encryption : EncryptionService
            The implementation of the encryption contract.
        """
        self._session_factory = session_factory
        self._encryption = encryption

    async def save(self, user_id: str, profile: dict):
        """
        Serialize, encrypt, and persist a user profile.

        Parameters
        ----------
        user_id : str
            The unique identifier for the user.
        profile : dict
            The dictionary of profile attributes to persist.
        """
        # Serialize and convert to bytes for encryption
        raw = json.dumps(profile).encode()
        encrypted = self._encryption.encrypt(raw)

        async with self._session_factory() as session:
            row = UserProfileORM(
                user_id=user_id,
                profile_encrypted=encrypted,
            )
            # Merge performs an update if the primary key (user_id) exists
            await session.merge(row)
            await session.commit()

    async def load(self, user_id: str):
        """
        Retrieve and decrypt a user profile from the database.

        Parameters
        ----------
        user_id : str
            The unique identifier of the user.

        Returns
        -------
        dict or None
            The decrypted profile as a dictionary, or None if the 
            user does not exist.
        """
        async with self._session_factory() as session:
            row = await session.get(UserProfileORM, user_id)
            if not row:
                return None
            
            # Decrypt bytes and parse JSON back to dictionary
            raw = self._encryption.decrypt(row.profile_encrypted)
            return json.loads(raw)


class SQLAlchemyDocumentHistoryRepository(DocumentHistoryRepository):
    """
    SQLAlchemy-based repository for tracking and limiting document history.
    """

    def __init__(self, config: DocumentHistoryConfig, session_factory):
        """
        Initialize the history repository and detect the database dialect.

        Parameters
        ----------
        config : DocumentHistoryConfig
            Configuration containing the history size limit.
        session_factory : async_sessionmaker
            The session factory bound to the target database.
        """
        self.max_history_size = config.history_size
        self._session_factory = session_factory

        engine = session_factory.kw["bind"]

        # Dialect-specific INSERT with upsert support
        if engine.dialect.name == "sqlite":
            self._insert = sqlite.insert
        elif engine.dialect.name == "postgresql":
            self._insert = postgresql.insert
        else:
            raise RuntimeError(f"Unsupported DB dialect: {engine.dialect.name}")

    async def log_sent_documents(self, user_id: str, documents: list[dict]):
        """
        Log document interactions and prune history beyond the size limit.

        Parameters
        ----------
        user_id : str
            The identifier of the user receiving documents.
        documents : list[dict]
            A list of document dictionaries containing 'document_id' and 'title'.
        """
        if not documents:
            return

        async with self._session_factory() as session:
            for doc in documents:
                # Prepare base insert statement
                stmt = self._insert(DocumentORM).values(
                    user_id=user_id,
                    document_id=doc["document_id"],
                    title=doc["title"],
                    sent_count=1,
                    sent_at=datetime.utcnow(),
                )

                stmt = stmt.on_conflict_do_update(
                    index_elements=["user_id", "document_id"],
                    set_={
                        "title": doc["title"],
                        "sent_count": DocumentORM.sent_count + 1,
                        "sent_at": datetime.utcnow(),
                    },
                )

                await session.execute(stmt)

            # Retrieve all document IDs for the user, ordered by most recent
            result = await session.execute(
                select(DocumentORM.id)
                .where(DocumentORM.user_id == user_id)
                .order_by(DocumentORM.sent_at.desc())
            )
            ids = [row[0] for row in result.all()]

            # Delete the oldest records if the count exceeds max_history_size
            if len(ids) > self.max_history_size:
                await session.execute(
                    delete(DocumentORM).where(DocumentORM.id.in_(ids[self.max_history_size :]))
                )

            await session.commit()

    async def get_user_history(self, user_id: str) -> list[dict]:
        """
        Retrieve the document interaction history for a given user.

        Parameters
        ----------
        user_id : str
            The unique identifier of the user.

        Returns
        -------
        list[dict]
            A list of document history records (mappings).
        """
        async with self._session_factory() as session:
            stmt = select(
                DocumentORM.document_id,
                DocumentORM.title,
                DocumentORM.sent_count,
            ).where(DocumentORM.user_id == user_id)

            result = await session.execute(stmt)
            # Returns a list of dictionaries via SQLAlchemy RowMapping
            return result.mappings().all()
