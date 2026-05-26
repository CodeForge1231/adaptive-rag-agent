from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, LargeBinary, String, UniqueConstraint

from .base import Base


class UserProfileORM(Base):
    """
    SQLAlchemy ORM model for storing encrypted user profiles.

    Attributes
    ----------
    user_id : sqlalchemy.Column
        The primary key representing a unique user identifier.
    profile_encrypted : sqlalchemy.Column
        The binary payload containing the encrypted profile data.
    updated_at : sqlalchemy.Column
        The UTC timestamp recorded when the profile was created or 
        last modified.
    """

    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True)
    profile_encrypted = Column(LargeBinary, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentORM(Base):
    """
    SQLAlchemy ORM model for tracking document interactions per user.

    Attributes
    ----------
    id : sqlalchemy.Column
        An auto-incrementing integer serving as the primary key.
    user_id : sqlalchemy.Column
        The identifier of the user who received the document.
    document_id : sqlalchemy.Column
        The unique identifier of the document retrieved from the store.
    title : sqlalchemy.Column
        The title or header of the document for easier identification.
    sent_at : sqlalchemy.Column
        The UTC timestamp of the first or most recent interaction.
    sent_count : sqlalchemy.Column
        A counter tracking the number of times this document was sent 
         to the user.
    """

    __tablename__ = "sent_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    document_id = Column(String, index=True)
    title = Column(String, index=True)
    sent_at = Column(DateTime, default=datetime.utcnow)
    sent_count = Column(Integer, default=1, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "document_id", name="_user_document_uc"),)
