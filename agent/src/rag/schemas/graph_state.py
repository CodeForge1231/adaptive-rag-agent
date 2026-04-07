from typing import List, TypedDict, Set

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class UserContext(TypedDict):
    """Represents the session-specific information about the current user."""
    
    user_id: str
    """Unique identifier for the user."""
    
    profile: dict
    """Dictionary containing user preferences, history, or metadata."""


class RAGState(TypedDict):
    """
    Main state object for the RAG agentic workflow.
    Tracks the progression of a query from input to final generation.
    """

    question: str
    """The original raw input query from the user."""
    
    messages: List[BaseMessage]
    """Full chat history including system messages, tool calls, and AI responses."""

    route: str
    """Target destination for the query (e.g., 'search', 'direct_answer', 'clarify')."""
    
    clarified: bool
    """Flag indicating whether the initial query was ambiguous and has been clarified."""
    
    missing_info: List[str]
    """List of specific details required from the user to proceed with retrieval."""

    rewritten_query: str
    """The optimized version of the user question for better search performance."""
    
    retriever_metadata: dict
    """Configuration or parameters used for the retrieval engine (e.g., k-values, thresholds)."""
    
    documents: List[Document]
    """The collection of document chunks retrieved for the current iteration."""
    
    documents_history: List[dict]
    """Logs of previously retrieved documents to avoid redundancy or track relevance."""

    retrieval_feedback: str  # "negative" | "neutral"
    """Evaluation of the retrieved documents relevance to the query."""
    
    feedback_action: str  # "rewrite" | "clarify"
    """The corrective strategy chosen based on retrieval feedback."""

    user: UserContext
    """User-specific data injected into the state for personalized processing."""
    
    answer: str
    """The final generated response or synthesis provided to the user."""
    
    sources: Set[str]
    """Set of unique source identifiers (URLs, paths) used to construct the answer."""