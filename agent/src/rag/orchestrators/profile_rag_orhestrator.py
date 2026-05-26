from langgraph.graph import END, START, StateGraph

from src.rag.schemas.graph_state import RAGState

from .base import BaseOrchestrator, OrchestratorRequirement


class ProfileRAGOrchestrator(BaseOrchestrator):
    """
    RAG orchestrator that integrates persistent user profiling into the workflow.

    Attributes
    ----------
    REQUIRED : set[OrchestratorRequirement]
        Functional requirements for this orchestrator, specifically PERSISTENCE.
    nodes : Any
        Implementation provider for the graph's functional nodes.
    graph : langgraph.graph.state.CompiledStateGraph
        The compiled state machine incorporating profile-aware logic.
    """
    REQUIRED = {OrchestratorRequirement.PERSISTENCE}

    def __init__(self, nodes):
        """
        Initialize the profile-aware orchestrator.

        Parameters
        ----------
        nodes : Any
            The provider of functional logic for all graph nodes.
        """
        self.nodes = nodes
        # Build and compile the graph during initialization
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Construct and compile the profile-aware RAG workflow..

        Returns
        -------
        langgraph.graph.state.CompiledStateGraph
            A compiled graph instance ready to process `RAGState`.
        """
        builder = StateGraph(RAGState)

        # Registration of message and profile handling nodes
        builder.add_node("add_user_message", self.nodes.add_user_message)
        builder.add_node("load_user_profile", self.nodes.load_user_profile)

        # Routing and transformation nodes
        builder.add_node("route", self.nodes.route_request)
        builder.add_node("rewrite", self.nodes.rewrite_query)

        # Contextual processing nodes
        builder.add_node("resolve_context", self.nodes.resolve_context)

        # Clarification logic nodes
        builder.add_node("check_info", self.nodes.check_clarification)
        builder.add_node("ask_clarify", self.nodes.ask_clarifying_question)

        # Retrieval and generation nodes
        builder.add_node("retrieve", self.nodes.retrieve_docs)
        builder.add_node("rerank", self.nodes.rerank_docs)
        builder.add_node("generate", self.nodes.generate)
        builder.add_node("simple", self.nodes.simple_response)

        # Finalization nodes
        builder.add_node("add_assistant_message", self.nodes.add_assistant_message)
        builder.add_node("update_profile", self.nodes.update_profile)

        # Initial flow
        builder.add_edge(START, "add_user_message")
        builder.add_edge("add_user_message", "load_user_profile")
        builder.add_edge("load_user_profile", "route")

        # Routing logic
        builder.add_conditional_edges(
            "route",
            lambda s: s["route"],
            {
                "simple_response": "simple",
                "vectorstore": "rewrite",
            },
        )

        # Intent extraction and context resolution
        builder.add_edge("rewrite", "resolve_context")

        # Clarification decision point after profile-aware context is resolved
        builder.add_edge("resolve_context", "check_info")

        builder.add_conditional_edges(
            "check_info",
            lambda s: "enough" if s["clarified"] else "missing",
            {
                "missing": "ask_clarify",
                "enough": "retrieve",
            },
        )

        # Non-retrieval paths converge to assistant message
        builder.add_edge("ask_clarify", "add_assistant_message")
        builder.add_edge("simple", "add_assistant_message")

        # RAG pipeline execution
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "generate")
        builder.add_edge("generate", "add_assistant_message")

        # Profile update sequence and termination
        builder.add_edge("add_assistant_message", "update_profile")
        builder.add_edge("update_profile", END)

        return builder.compile()
