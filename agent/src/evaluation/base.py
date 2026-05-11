from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEvaluator(ABC):
    """Abstract base class for RAG evaluation."""

    @abstractmethod
    async def start_retrieval_tests(self, retriever, max_concurrency: int = 10) -> List[dict]:
        """
        Run evaluation for the retrieval component.

        Parameters
        ----------
        retriever : BaseRetriever
            The retrieval component instance to be tested.
        max_concurrency : int, default 10
            The maximum number of concurrent asynchronous retrieval tasks.

        Returns
        -------
        List[dict]
            A list of results containing metrics like recall, MRR, or precision.
        """
        pass

    @abstractmethod
    async def start_answer_tests(
        self, orchestrator, base_state: Dict[str, Any], max_concurrency: int = 5
    ) -> List[dict]:
        """
        Run evaluation for the generation (orchestration) component.

        Parameters
        ----------
        orchestrator : RAGOrchestrator
            The full pipeline instance to generate answers.
        base_state : Dict[str, Any]
            The initial state or context required for the orchestrator.
        max_concurrency : int, default 5
            The maximum number of concurrent generation requests.

        Returns
        -------
        List[dict]
            A list of evaluation results, typically including faithfulness, 
            relevance, or LLM-assisted metrics.
        """
        pass
