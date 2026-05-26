from langgraph.graph import END, START, StateGraph

from src.rag.schemas.graph_state import RAGState

from .base import BaseOrchestrator


class RerankingRAGOrchestrator(BaseOrchestrator):
    """
    Orchestrator for a RAG pipeline with an explicit document reranking stage.

    This class defines a workflow that fetches documents from a vector store 
    and applies a reranking model to prioritize the most relevant context 
    before passing it to the generator. It includes routing and 
    clarification logic to handle various user intents.

    Attributes
    ----------
    nodes : Any
        Implementation provider for the functional nodes used in the graph.
    graph : langgraph.graph.state.CompiledStateGraph
        The compiled state machine representing the reranking-enabled workflow.
    """
    def __init__(self, nodes):
        """
        Initialize the reranking orchestrator and compile its graph.

        Parameters
        ----------
        nodes : Any
            The provider of functional logic for all graph nodes.
        """
        self.nodes = nodes
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Construct and compile the LangGraph workflow with reranking.

        Returns
        -------
        langgraph.graph.state.CompiledStateGraph
            A compiled graph instance optimized for RAG operations with reranking.
        """
        builder = StateGraph(RAGState)

        # Basic interaction and routing nodes
        builder.add_node("add_user_message", self.nodes.add_user_message)
        builder.add_node("add_assistant_message", self.nodes.add_assistant_message)
        builder.add_node("route", self.nodes.route_request)

        # Clarification and transformation nodes
        builder.add_node("check_info", self.nodes.check_clarification)
        builder.add_node("ask_clarify", self.nodes.ask_clarifying_question)
        builder.add_node("rewrite", self.nodes.rewrite_query)

        # Retrieval, reranking, and generation nodes
        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("generate", self.nodes.generate)
        builder.add_node("simple", self.nodes.simple_response)
        builder.add_node("rerank_docs", self.nodes.rerank_docs)

        # Initial execution flow
        builder.add_edge(START, "add_user_message")
        builder.add_edge("add_user_message", "route")

        # Routing
        builder.add_conditional_edges(
            "route",
            lambda s: s["route"],
            {
                "simple_response": "simple",
                "vectorstore": "check_info",
            },
        )

        # Clarification logic to ensure query specificity
        builder.add_conditional_edges(
            "check_info",
            lambda s: "enough" if s["clarified"] else "missing",
            {
                "missing": "ask_clarify",
                "enough": "rewrite",
            },
        )

        # Core RAG sequence:
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "rerank_docs")
        builder.add_edge("rerank_docs", "generate")

        # Consolidation of all paths to response finalization
        builder.add_edge("generate", "add_assistant_message")
        builder.add_edge("ask_clarify", "add_assistant_message")
        builder.add_edge("simple", "add_assistant_message")

        builder.add_edge("add_assistant_message", END)
        
        return builder.compile()
