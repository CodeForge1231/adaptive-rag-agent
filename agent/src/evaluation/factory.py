from .evaluator import RAGEvaluator


class EvaluatorFactory:
    """
    Factory class for initializing RAG evaluation components.
    """
    
    @staticmethod
    def create(judge_llm, config: dict) -> RAGEvaluator:
        """
        Create and return an instance of RAGEvaluator.

        Parameters
        ----------
        judge_llm : BaseChatModel
            The language model instance used to judge and score responses.
        config : dict
            Configuration dictionary containing evaluation parameters 
            and file paths.

        Returns
        -------
        RAGEvaluator
            A configured evaluator instance ready for testing.
        """
        return RAGEvaluator(judge_llm, config)
