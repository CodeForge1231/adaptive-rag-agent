from typing import List
from pydantic import BaseModel, Field


class MessageSchema(BaseModel):
    """
    Schema representing a single chat message.

    Attributes
    ----------
    role : str
        The entity role (e.g., 'user', 'assistant').
    content : str
        The textual content of the message.
    """
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class UserSchema(BaseModel):
    """
    Schema for user identification data.

    Attributes
    ----------
    user_id : str
        Unique identifier for the user.
    """
    user_id: str = Field(..., description="Unique user identifier")


class AskRequest(BaseModel):
    """
    Request schema for asking a question.

    Attributes
    ----------
    question : str
        The user's query.
    messages : List[MessageSchema]
        Conversation history.
    user : UserSchema
        User context information.
    """
    question: str = Field(..., description="The user's question")
    messages: List[MessageSchema] = Field(default=[], description="Chat history")
    user: UserSchema = Field(..., description="User identification")


class AskResponse(BaseModel):
    """
    Response schema for a chat query.

    Attributes
    ----------
    answer : str
        The generated assistant response.
    messages : List[MessageSchema]
        Updated conversation history.
    documents : List
        Source documents used for retrieval.
    """
    answer: str = Field(..., description="The generated answer")
    messages: List[MessageSchema] = Field(..., description="Updated chat history")
    documents: List = Field(default=[], description="Retrieved source documents")