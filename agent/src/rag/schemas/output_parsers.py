from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    """Container for user-specific attributes used to filter and personalize RAG results."""
    
    age_group: Optional[Literal["teen", "adult", "senior"]] = Field(
        None, 
        description="Categorization of the user based on age to adjust response complexity and tone"
    )
    level: Optional[Literal["beginner", "intermediate", "advanced"]] = Field(
        None, 
        description="The technical proficiency level of the user in AI/ML domains"
    )
    topic: Optional[Literal["python", "ml", "dl", "nlp"]] = Field(
        None, 
        description="The specific subject area relevant to the current user inquiry"
    )


class RetrieverIntent(BaseModel):
    """Represents the processed intent for the retrieval engine."""
    
    retriever_query: str = Field(
        description="Refined and optimized semantic query string for vector database search"
    )
    metadata: Metadata = Field(
        description="Filter criteria and personalization context for the retrieval process"
    )


class RouterDecision(BaseModel):
    """Decision output determining the execution path for the incoming query."""
    
    datasource: Literal["vectorstore", "simple_response"] = Field(
        description="The selected data source: 'vectorstore' for RAG or 'simple_response' for direct LLM generation"
    )


class ClarificationDecision(BaseModel):
    """Evaluation of whether the provided information is sufficient to proceed."""
    
    result: Literal["ENOUGH", "MISSING"] = Field(
        description="Status indicating if the context is complete or requires additional user input"
    )
    missing: List[Literal["age_group", "level", "topic"]] | None = Field(
        None, 
        description="A collection of specific metadata fields that must be provided by the user"
    )


class RerankItem(BaseModel):
    """Individual document ranking entry after post-retrieval evaluation."""
    
    index: int = Field(
        description="Original position or identifier of the document in the retrieved list"
    )
    score: int = Field(
        ge=0, le=10, 
        description="Relevance score assigned to the document, where 10 is highly relevant and 0 is irrelevant"
    )


class RerankResult(BaseModel):
    """Collection of ranked items representing the final document ordering."""
    
    rankings: List[RerankItem] = Field(
        description="Ordered list of documents sorted by their calculated relevance scores"
    )


class FeedbackDecision(BaseModel):
    """Evaluation of the retrieval quality or agent performance."""
    
    feedback: Literal["negative", "neutral"] = Field(
        description="Qualitative assessment of the retrieved information's utility"
    )


class FeedbackAction(BaseModel):
    """Correction strategy to be executed in response to sub-optimal feedback."""
    
    action: Literal["rewrite", "clarify"] = Field(
        description="Specific recovery step: 'rewrite' for query optimization or 'clarify' for user interrogation"
    )