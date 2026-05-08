from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser

from src.rag.schemas.output_parsers import (
    ClarificationDecision,
    FeedbackAction,
    FeedbackDecision,
    RerankResult,
    RetrieverIntent,
    RouterDecision,
)

from .prompts import (
    CLARIFICATION_PROMPT,
    CLARIFICATION_QUESTION_PROMPT,
    FEEDBACK_CLARIFICATION_PROMPT,
    FEEDBACK_DECISION_PROMPT,
    NEGATIVE_FEEDBACK_PROMPT,
    PROFILE_UPDATER_PROMPT,
    RAG_RESPONSE_PROMPT,
    RERANK_PROMPT,
    REWRITE_PROMPT,
    REWRITE_WITH_FEEDBACK_PROMPT,
    ROUTER_PROMPT,
    SIMPLE_RESPONSE_PROMPT,
)

REWRITE_PROMPT_PARSER = PydanticOutputParser(pydantic_object=RetrieverIntent)
ROUTER_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=RouterDecision)
CLARIFICATION_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=ClarificationDecision)
RERANK_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=RerankResult)
NEGATIVE_FEEDBACK_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=FeedbackDecision)
FEEDBACK_ACTION_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=FeedbackAction)


class ChainLibrary:
    """
    A centralized library of LangChain Runnables for the RAG agent.
    
    This class orchestrates the assembly of prompts, LLMs, and output parsers 
    into functional chains for routing, retrieval, generation, and self-correction.
    """

    def __init__(self, fast_llm, heavy_llm):
        """
        Initialize the library with specific LLM runnables.

        Args:
            fast_llm: A cost-efficient, low-latency model for classification and parsing tasks.
            heavy_llm: A high-capacity model for complex reasoning and final answer synthesis.
        """
        self.fast_llm = fast_llm
        self.heavy_llm = heavy_llm

    @property
    def router(self):
        """Chain to determine whether to use the vector store or provide a direct response."""
        return ROUTER_PROMPT | self.fast_llm.runnable | ROUTER_OUTPUT_PARSER

    @property
    def clarification_checker(self):
        """Chain to evaluate if the current user query has sufficient context for retrieval."""
        return CLARIFICATION_PROMPT | self.fast_llm.runnable | CLARIFICATION_OUTPUT_PARSER

    @property
    def clarification_asker(self):
        """Chain to generate a natural language question requesting missing information from the user."""
        return CLARIFICATION_QUESTION_PROMPT | self.fast_llm.runnable | StrOutputParser()

    @property
    def generate(self):
        """Chain for high-quality RAG synthesis using retrieved documents and reasoning."""
        return RAG_RESPONSE_PROMPT | self.heavy_llm.runnable | StrOutputParser()

    @property
    def simple_generator(self):
        """Chain for generating quick responses to general inquiries without document retrieval."""
        return SIMPLE_RESPONSE_PROMPT | self.fast_llm.runnable | StrOutputParser()

    @property
    def rewriter(self):
        """Chain for transforming a user question into an optimized search query for vector stores."""
        return REWRITE_PROMPT | self.fast_llm.runnable | REWRITE_PROMPT_PARSER

    @property
    def profile_updater(self):
        """Chain to extract and update user metadata and preferences in JSON format."""
        return PROFILE_UPDATER_PROMPT | self.fast_llm.runnable | JsonOutputParser()

    @property
    def reranker(self):
        """Chain to score and re-rank retrieved documents based on their actual relevance to the query."""
        return RERANK_PROMPT | self.fast_llm.runnable | RERANK_OUTPUT_PARSER

    @property
    def feedback_detector(self):
        """Chain to analyze the retrieval stage and detect potential issues or lack of information."""
        return NEGATIVE_FEEDBACK_PROMPT | self.fast_llm.runnable | NEGATIVE_FEEDBACK_OUTPUT_PARSER

    @property
    def feedback_decision_maker(self):
        """Chain to select the best corrective action after sub-optimal retrieval is detected."""
        return FEEDBACK_DECISION_PROMPT | self.fast_llm.runnable | FEEDBACK_ACTION_OUTPUT_PARSER

    @property
    def feedback_clarifier(self):
        """Chain to ask for clarification specifically when retrieval feedback suggests ambiguity."""
        return FEEDBACK_CLARIFICATION_PROMPT | self.fast_llm.runnable | StrOutputParser()

    @property
    def history_rewriter(self):
        """Chain to rewrite queries by incorporating previous conversation context and feedback loops."""
        return REWRITE_WITH_FEEDBACK_PROMPT | self.fast_llm.runnable | StrOutputParser()