from fastapi import APIRouter, Depends

from src.api.deps import get_context, verify_token
from src.api.schemas.chat import AskRequest, AskResponse
from src.api.use_cases.ask_question import AskQuestionUseCase

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_token)])

async def ask(data: AskRequest, ctx=Depends(get_context)):
    """
    Handle a user query through the RAG orchestrator.

    This endpoint receives a question and conversation history, processes 
    it using the retrieval-augmented generation pipeline, and returns 
    the assistant's response.

    Parameters
    ----------
    data : AskRequest
        The request body containing the user's question, chat history, 
        and optional configuration.
    ctx : AppContext
        Application context injected via dependency, providing access 
        to the RAG orchestrator.

    Returns
    -------
    AskResponse
        A structured response containing the generated answer and 
        updated message history.

    Raises
    ------
    HTTPException
        If the token verification fails or an internal error occurs 
        during execution.
    """
    # Initialize the specific business logic use case
    use_case = AskQuestionUseCase(ctx.orchestrator)

    # Execute the request processing and return the result
    result = await use_case.execute(data)

    return result
