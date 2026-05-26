from typing import Dict, Any
from pydantic import BaseModel, Field


class AddDocumentRequest(BaseModel):
    """
    Request schema for adding a document to the vector store.

    Attributes
    ----------
    id : str
        Unique identifier for the document.
    text : str
        Main content of the document.
    metadata : Dict[str, Any]
        Associated document attributes.
    """
    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Raw text content")
    metadata: Dict[str, Any] = Field(..., description="Key-value metadata pairs")


class AddDocumentResponse(BaseModel):
    """
    Response schema for document ingestion confirmation.

    Attributes
    ----------
    success : bool
        Indicates if the operation was successful.
    """
    success: bool = Field(..., description="Ingestion status flag")