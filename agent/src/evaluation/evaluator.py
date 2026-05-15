import asyncio
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from tqdm.asyncio import tqdm as tqdm_async

from src.core.observability import observability

from .base import BaseEvaluator
from .prompts import ANSWER_JUDGE_PROMPT
from .schemas import AnswerEval, RetrievalEval, TestQuestion


class RAGEvaluator(BaseEvaluator):
    """
    Evaluator for RAG systems providing retrieval and generation benchmarks.

    Attributes
    ----------
    config : dict
        Configuration settings for the evaluator.
    chain : Runnable
        LangChain sequence for judging answers using an LLM.
    tests : List[TestQuestion]
        Loaded test cases for evaluation.
    answer_results : List[Dict]
        Stored results of answer generation tests.
    retrieval_results : List[Dict]
        Stored results of document retrieval tests.
    """

    def __init__(self, judge_llm, config):
        """
        Initialize the evaluator with a judge LLM and settings.
        """
        self.config = config
        self.chain = (
            ANSWER_JUDGE_PROMPT | judge_llm.runnable | JsonOutputParser(pydantic_object=AnswerEval)
        )
        self.tests: List[TestQuestion] = []
        self.answer_results: List[Dict] = []

        self.retrieval_results: List[Dict] = []

    def _convert_history_to_messages(self, history: List[dict]) -> List[BaseMessage]:
        """
        Converts dict-style history to LangChain message objects.

        Parameters
        ----------
        history : List[dict]
            List of messages with 'role' and 'content' keys.

        Returns
        -------
        List[BaseMessage]
            Converted HumanMessage and AIMessage objects.
        """
        messages = []
        for entry in history:
            role = entry.get("role")
            content = entry.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role in ["assistant", "ai"]:
                messages.append(AIMessage(content=content))
        return messages

    @observability.time_it("Calculate MRR score")
    def _calculate_mrr(self, keyword: str, retrieved_docs: list) -> float:
        """
        Calculate reciprocal rank for a single keyword.

        Parameters
        ----------
        keyword : str
            The target keyword to find in retrieved documents.
        retrieved_docs : list
            List of documents returned by the retriever.

        Returns
        -------
        float
            Reciprocal rank value (1/rank) or 0.0 if not found.
        """
        keyword_lower = keyword.lower()
        for rank, doc in enumerate(retrieved_docs, start=1):
            if keyword_lower in doc.page_content.lower():
                return 1.0 / rank
        return 0.0

    @observability.time_it("Calculate DCG score")
    def _calculate_dcg(self, relevances: list[int]) -> float:
        """
        Calculate Discounted Cumulative Gain.

        Parameters
        ----------
        relevances : list[int]
            List of relevance scores (e.g., 0 or 1).

        Returns
        -------
        float
            The DCG score for the ranked list.
        """
        dcg = 0.0
        for i, rel in enumerate(relevances):
            dcg += rel / math.log2(i + 2)
        return dcg

    @observability.time_it("Calculate nDCG score")
    def _calculate_ndcg(self, keyword: str, retrieved_docs: list) -> float:
        """
        Calculate nDCG for a single keyword with binary relevance.

        Parameters
        ----------
        keyword : str
            The target keyword.
        retrieved_docs : list
            List of retrieved documents.

        Returns
        -------
        float
            Normalized DCG score between 0.0 and 1.0.
        """
        keyword_lower = keyword.lower()

        if not retrieved_docs:
            return 0.0

        relevances = [
            1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs
        ]

        dcg = self._calculate_dcg(relevances)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self._calculate_dcg(ideal_relevances)

        return dcg / idcg if idcg > 0 else 0.0

    @observability.time_it_async("Single retrieval evaluation")
    async def _evaluate_retrieval_task(
        self, retriever, test: TestQuestion, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval for a single question with concurrency control.

        Parameters
        ----------
        retriever : BaseRetriever
            The retriever instance to test.
        test : TestQuestion
            The test case containing the question and expected keywords.
        semaphore : asyncio.Semaphore
            Semaphore to limit parallel execution.

        Returns
        -------
        Dict[str, Any]
            Calculated metrics and status for the retrieval task.
        """
        async with semaphore:
            try:
                retrieved_docs = await retriever.aretrieve(test.question)

                mrr_scores = [self._calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
                ndcg_scores = [self._calculate_ndcg(kw, retrieved_docs) for kw in test.keywords]

                found = sum(1 for s in mrr_scores if s > 0)
                total = len(test.keywords)

                metrics = RetrievalEval(
                    mrr=sum(mrr_scores) / total if total > 0 else 0.0,
                    ndcg=sum(ndcg_scores) / total if total > 0 else 0.0,
                    keywords_found=found,
                    total_keywords=total,
                    keyword_coverage=(found / total * 100) if total > 0 else 0.0,
                )

                return {
                    "question": test.question,
                    "category": test.category,
                    "status": "success",
                    **metrics.model_dump(),
                }
            except Exception as e:
                return {
                    "question": test.question,
                    "category": test.category,
                    "status": "failed",
                    "error": str(e),
                }

    @observability.time_it_async("Single answer evaluation")
    async def _evaluate_answer(self, orchestrator, test, base_state, semaphore):
        """
        Evaluate a single answer using the orchestrator and judge LLM.

        Parameters
        ----------
        orchestrator : RAGOrchestrator
            The pipeline to generate answers.
        test : TestQuestion
            The test case data.
        base_state : dict
            Template for the orchestrator's state.
        semaphore : asyncio.Semaphore
            Semaphore for parallel task limiting.

        Returns
        -------
        dict
            Evaluation metrics (accuracy, relevance, etc.) from the judge.
        """
        async with semaphore:
            try:
                # Clean copy of state
                current_state = copy.deepcopy(base_state)

                # History conversion
                test_history = getattr(test, "history", [])
                if test_history:
                    current_state["messages"] = self._convert_history_to_messages(test_history)

                # Set current question
                current_state["question"] = test.question

                # Run orchestrator
                final_state = await orchestrator.run(current_state)
                generated_answer = final_state.get("answer", "")

                # Judge Evaluation
                # Don't forget to include format_instructions if needed
                evaluation = await self.chain.ainvoke(
                    {
                        "question": test.question,
                        "generated_answer": generated_answer,
                        "reference_answer": test.reference_answer,
                        "format_instructions": self.chain.last.get_format_instructions(),
                    }
                )

                return {
                    "question": test.question,
                    "category": test.category,
                    "generated_answer": generated_answer,
                    "reference_answer": test.reference_answer,
                    "accuracy": evaluation.get("accuracy"),
                    "completeness": evaluation.get("completeness"),
                    "relevance": evaluation.get("relevance"),
                    "history_depth": len(test_history),
                    "status": "success",
                }
            except Exception as e:
                print(f"Error on question '{test.question}': {str(e)}")
                return {"question": test.question, "status": "failed", "error": str(e)}

    def load_tests(self, file_path: Optional[str] = None):
        """
        Load test cases from a JSON file.

        Parameters
        ----------
        file_path : str, optional
            Path to the JSON file. Defaults to config value.

        Raises
        ------
        ValueError
            If no path is provided or configured.
        FileNotFoundError
            If the file does not exist.
        """
        path = file_path or self.config.get("app").get("evaluation", {}).get("test_file")

        if not path:
            raise ValueError("Test file path must be provided or set in config.")

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Test file not found at: {path_obj}")

        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.tests = [TestQuestion(**item) for item in data]
        print(f"Loaded {len(self.tests)} tests into RAGEvaluator")

    @observability.time_it_async("Full retrieval test suite")
    async def start_retrieval_tests(self, retriever, max_concurrency: int = 10) -> List[dict]:
        """
        Run retrieval evaluation for all tests in parallel.

        Parameters
        ----------
        retriever : BaseRetriever
            The retriever component to test.
        max_concurrency : int, default 10
            Max parallel retrieval requests.

        Returns
        -------
        List[dict]
            Aggregate results for the retrieval suite.
        """
        if not self.tests:
            print("No tests loaded in RAGEvaluator. Call load_tests() first.")
            return []

        self.retrieval_results = []

        semaphore = asyncio.Semaphore(max_concurrency)

        tasks = [self._evaluate_retrieval_task(retriever, test, semaphore) for test in self.tests]

        results = await tqdm_async.gather(*tasks, desc="Parallel Retrieval Evaluation")

        self.retrieval_results = results

        print(f"Retrieval evaluation finished. Processed {len(self.retrieval_results)} tests.")
        return self.retrieval_results

    @observability.time_it_async("Full answer test suite")
    async def start_answer_tests(
        self, orchestrator, base_state: Dict[str, Any], max_concurrency: int = 5
    ) -> List[dict]:
        """
        Run answer generation and judging in parallel.

        Parameters
        ----------
        orchestrator : RAGOrchestrator
            The pipeline to test.
        base_state : dict
            Initial state template.
        max_concurrency : int, default 5
            Max parallel generation/judging tasks.

        Returns
        -------
        List[dict]
            Aggregate results for the answer suite.
        """
        if not self.tests:
            return []

        # Create a semaphore to limit parallel tasks
        semaphore = asyncio.Semaphore(max_concurrency)

        # Create a list of tasks
        tasks = [
            self._evaluate_answer(orchestrator, test, base_state, semaphore) for test in self.tests
        ]

        # Use tqdm.asyncio.gather to show progress bar for parallel tasks
        results = await tqdm_async.gather(*tasks, desc="Parallel Answer Evaluation")

        self.answer_results = results
        print(f"Finished. Processed {len(self.answer_results)} tests.")
        return self.answer_results
