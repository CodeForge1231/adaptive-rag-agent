from langgraph.graph import END, START, StateGraph

from src.rag.schemas.graph_state import RAGState

from .base import BaseOrchestrator, OrchestratorRequirement
from .subgraphs.profile_update_subgraph import ProfileUpdateSubgraph
from .subgraphs.retrieval_subgraph import RetrievalSubgraph


class FeedbackAwareRAGOrchestrator(BaseOrchestrator):
    """
    Advanced RAG orchestrator with feedback detection and history awareness.

    Attributes
    ----------
    REQUIRED : set[OrchestratorRequirement]
        Functional requirements for this orchestrator, including PERSISTENCE.
    profile_subgraph : ProfileUpdateSubgraph
        Encapsulated logic for updating user profile information.
    retrieval_subgraph : RetrievalSubgraph
        Encapsulated logic for document retrieval processes.
    nodes : Any
        Implementation provider for all graph nodes.
    graph : langgraph.graph.state.CompiledStateGraph
        The compiled state machine for the feedback-aware RAG process.
    """

    REQUIRED = {OrchestratorRequirement.PERSISTENCE}

    def __init__(self, nodes):
        """
        Initialize the feedback-aware orchestrator and its subgraphs.

        Parameters
        ----------
        nodes : Any
            The provider of functional logic for all graph nodes.
        """
        self.profile_subgraph = ProfileUpdateSubgraph(nodes)
        self.retrieval_subgraph = RetrievalSubgraph(nodes)
        super().__init__(nodes)

    def _build_graph(self):
        """
        Construct and compile the feedback-aware RAG workflow.

        The graph features multiple conditional entry points and specialized 
        paths for handling neutral or negative retrieval feedback.

        Returns
        -------
        langgraph.graph.state.CompiledStateGraph
            A compiled graph instance capable of processing `RAGState`.
        """
        builder = StateGraph(RAGState)

        # Basic interaction nodes
        builder.add_node("add_user_message", self.nodes.add_user_message)
        builder.add_node("route", self.nodes.route_request)
        builder.add_node("check_info", self.nodes.check_clarification)
        builder.add_node("ask_clarify", self.nodes.ask_clarifying_question)

        # Output and generation nodes
        builder.add_node("simple", self.nodes.simple_response)
        builder.add_node("add_assistant_message", self.nodes.add_assistant_message)
        builder.add_node("generate", self.nodes.generate)

        # Retrieval and history nodes
        builder.add_node("load_document_history", self.nodes.load_document_history)
        builder.add_node("rewrite", self.nodes.rewrite_query)
        builder.add_node("retrieve_initial", self.nodes.retrieve_docs)

        # Feedback processing nodes
        builder.add_node("detect_negative_feedback", self.nodes.detect_negative_feedback)
        builder.add_node("feedback_decision", self.nodes.feedback_decision)
        builder.add_node("ask_feedback_clarify", self.nodes.ask_feedback_clarify)
        builder.add_node("rewrite_with_history", self.nodes.rewrite_with_history)
        builder.add_node("retrieve_after_feedback", self.nodes.retrieve_with_exclusion)

        # Post-retrieval processing
        builder.add_node("rerank", self.nodes.rerank_docs)
        builder.add_node("log_documents", self.nodes.log_documents)

        # Execution flow: Entry and routing
        builder.add_edge(START, "add_user_message")
        builder.add_edge("add_user_message", "route")

        builder.add_conditional_edges(
            "route",
            lambda s: s["route"],
            {
                "simple_response": "simple",
                "vectorstore": "check_info",
            },
        )

        # Information sufficiency check
        builder.add_conditional_edges(
            "check_info",
            lambda s: "enough" if s["clarified"] else "missing",
            {
                "missing": "ask_clarify",
                "enough": "load_document_history",
            },
        )

        builder.add_edge("load_document_history", "detect_negative_feedback")

        # Feedback-based branching
        builder.add_conditional_edges(
            "detect_negative_feedback",
            lambda s: s.get("retrieval_feedback", "neutral"),
            {
                "neutral": "rewrite",
                "negative": "feedback_decision",
            },
        )

        builder.add_conditional_edges(
            "feedback_decision",
            lambda s: s.get("feedback_action"),
            {
                "rewrite": "rewrite_with_history",
                "clarify": "ask_feedback_clarify",
            },
        )

        # Standard vs history-aware retrieval paths
        builder.add_edge("rewrite", "retrieve_initial")
        builder.add_edge("retrieve_initial", "rerank")

        builder.add_edge("rewrite_with_history", "retrieve_after_feedback")
        builder.add_edge("retrieve_after_feedback", "rerank")

        # Convergence to generation
        builder.add_edge("rerank", "log_documents")
        builder.add_edge("log_documents", "generate")
        builder.add_edge("generate", "add_assistant_message")

        # Universal assistant message processing and exit
        builder.add_edge("simple", "add_assistant_message")
        builder.add_edge("ask_clarify", "add_assistant_message")
        builder.add_edge("ask_feedback_clarify", "add_assistant_message")

        builder.add_edge("add_assistant_message", END)

        return builder.compile()
