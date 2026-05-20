from fastapi import APIRouter, Depends

from src.api.deps import get_context, verify_token
from src.api.schemas.document import (
    AddDocumentRequest,
    AddDocumentResponse,
)
from src.api.use_cases.add_document import (
    AddDocumentUseCase,
)

# Define router for document-related operations
router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

@router.post(
    "/add",
    response_model=AddDocumentResponse,
    dependencies=[Depends(verify_token)],
)
async def add_document(
    data: AddDocumentRequest,
    ctx=Depends(get_context),
):
    """
    Register a new document in the system's vector storage.

    Parameters
    ----------
    data : AddDocumentRequest
        The schema containing document text and associated metadata.
    ctx : AppContext
        The application context providing access to the vectorstore instance.

    Returns
    -------
    AddDocumentResponse
        A response confirming the successful indexing of the document.

    Raises
    ------
    HTTPException
        If authentication fails or the document cannot be processed.
    """
    # Initialize the document ingestion logic with the configured vector store
    use_case = AddDocumentUseCase(
        ctx.vectorstore
    )

    # Execute the document registration process
    result = await use_case.execute(data)

    return result