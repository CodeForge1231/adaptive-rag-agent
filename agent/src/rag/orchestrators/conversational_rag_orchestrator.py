from langgraph.graph import END, START, StateGraph

from src.rag.schemas.graph_state import RAGState

from .base import BaseOrchestrator


class ConversationalRAGOrchestrator(BaseOrchestrator):
    """
    Orchestrator for a conversational RAG pipeline with clarification logic.

    Attributes
    ----------
    nodes : Any
        A collection of node implementations for routing, rewriting, 
        retrieval, and generation.
    graph : langgraph.graph.state.CompiledStateGraph
        The compiled state machine that executes the conversational RAG logic.
    """
    def __init__(self, nodes):
        """
        Initialize the orchestrator and build the conversational graph.

        Parameters
        ----------
        nodes : Any
            The provider of functional logic for all graph nodes.
        """
        self.nodes = nodes
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Construct and compile the conversational RAG workflow.

        The graph includes conditional routing to handle simple queries or 
        complex RAG tasks requiring potential clarification and rewriting.

        Returns
        -------
        langgraph.graph.state.CompiledStateGraph
            A compiled graph instance capable of processing `RAGState`.
        """
        builder = StateGraph(RAGState)

        # Register message handling and routing nodes
        builder.add_node("add_user_message", self.nodes.add_user_message)
        builder.add_node("add_assistant_message", self.nodes.add_assistant_message)
        builder.add_node("route", self.nodes.route_request)
        builder.add_node("check_info", self.nodes.check_clarification)
        builder.add_node("ask_clarify", self.nodes.ask_clarifying_question)
        builder.add_node("rewrite", self.nodes.rewrite_query)
        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("generate", self.nodes.generate)
        builder.add_node("simple", self.nodes.simple_response)

        # Define entry point and initial flow
        builder.add_edge(START, "add_user_message")
        builder.add_edge("add_user_message", "route")

        # Routing logic: direct response or RAG path
        builder.add_conditional_edges(
            "route",
            lambda s: s["route"],
            {
                "simple_response": "simple",
                "vectorstore": "check_info",
            },
        )

       # Clarification logic: ensure sufficient information exists for retrieval
        builder.add_conditional_edges(
            "check_info",
            lambda s: "enough" if s["clarified"] else "missing",
            {
                "missing": "ask_clarify",
                "enough": "rewrite",
            },
        )

        # Standard RAG execution flow
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "generate")

        # Convergence of all paths to state persistence and termination
        builder.add_edge("generate", "add_assistant_message")
        builder.add_edge("ask_clarify", "add_assistant_message")
        builder.add_edge("simple", "add_assistant_message")
        builder.add_edge("add_assistant_message", END)
        return builder.compile()
