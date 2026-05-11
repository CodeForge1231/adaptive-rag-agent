from pydantic import BaseModel, Field


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")
    history: list[dict]


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description=(
            "Detailed yet concise justification for the assigned scores. "
            "Explain exactly where the answer deviates from the reference or "
            "how the retrieved context supports/refutes the generated response."
        )
    )

    accuracy: float = Field(
        description=(
            "Factual correctness compared to the reference answer. "
            "1: Factually incorrect or contains hallucinations; "
            "3: Mostly correct but has minor inaccuracies; "
            "5: Perfectly accurate and verified by the context."
        )
    )
    completeness: float = Field(
        description=(
            "Coverage of all key points from the reference answer. "
            "1: Missing most critical information; "
            "3: Addresses the main point but skips secondary details; "
            "5: Includes every single piece of information from the reference answer."
        )
    )
    relevance: float = Field(
        description=(
            "Alignment with the user's specific question. "
            "1: Completely off-topic; "
            "3: Answers the question but includes redundant or distracting info; "
            "5: Laser-focused on the query with no irrelevant content."
        )
    )
