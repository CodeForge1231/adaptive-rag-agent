from langgraph.graph import END, START, StateGraph

from src.rag.schemas.graph_state import RAGState

from .base import BaseSubgraph


class RetrievalSubgraph(BaseSubgraph):
    """
    A subgraph responsible for document retrieval and reranking within a RAG pipeline.

    This class encapsulates the logic for fetching relevant documents from a 
    vector store and refining them using a reranking model to improve quality 
    before generation.

    Attributes
    ----------
    graph : langgraph.graph.state.CompiledStateGraph
        The compiled LangGraph executable representing the retrieval workflow.
    nodes : Any
        An object containing the implementation logic for 'retrieve_docs' 
        and 'rerank_docs'.
    """
    def __init__(self, nodes):
        super().__init__(nodes)
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Construct and compile the LangGraph workflow for document retrieval.

        The workflow starts with document fetching, followed by a reranking 
        process, and then terminates.

        Returns
        -------
        langgraph.graph.state.CompiledStateGraph
            A compiled graph ready to process `RAGState` objects.
        """
        builder = StateGraph(RAGState)

        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("rerank_docs", self.nodes.rerank_docs)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "rerank_docs")
        builder.add_edge("rerank_docs", END)

        return builder.compile()
