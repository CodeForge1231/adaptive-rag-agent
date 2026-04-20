from abc import ABC, abstractmethod

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    """

    pass


class UserProfileRepository(ABC):
    """
    Abstract repository for user profile persistence.
    """

    @abstractmethod
    async def load(self, user_id: str):
        """
        Retrieve a user profile from the data store by its identifier.

        Parameters
        ----------
        user_id : str
            The unique identifier of the user.

        Returns
        -------
        dict or None
            The user profile data if found, otherwise None.
        """
        pass

    @abstractmethod
    async def save(self, user_id: str, profile: dict):
        """
        Persist or update a user profile in the data store.

        Parameters
        ----------
        user_id : str
            The unique identifier of the user.
        profile : dict
            The profile data to be stored.
        """
        pass


class DocumentHistoryRepository(ABC):
    """
    Abstract repository for document history tracking.
    """

    @abstractmethod
    async def log_sent_documents(self, user_id: str, documents: list[dict]):
        """
        Record a list of documents provided to a user in a specific session.

        Parameters
        ----------
        user_id : str
            The unique identifier of the user.
        documents : list[dict]
            A collection of document metadata or content to be logged.
        """
        pass

    @abstractmethod
    async def get_user_history(self, user_id: str):
        """
        Fetch the complete document interaction history for a user.

        Parameters
        ----------
        user_id : str
            The unique identifier of the user.

        Returns
        -------
        list[dict]
            A list of previously logged document interactions.
        """
        pass
