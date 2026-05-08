from langgraph.graph import END, START, StateGraph

from src.rag.schemas.graph_state import RAGState

from .base import BaseOrchestrator


class SimpleOrchestrator(BaseOrchestrator):
    """
    A minimal orchestrator for handling direct user-to-assistant interactions.

    This class implements a linear LangGraph workflow without retrieval or 
    complex routing logic. It is designed for basic chat functionality where 
    input is processed into a simple response and stored in the message history.

    Attributes
    ----------
    nodes : Any
        Implementation provider for message handling and response generation nodes.
    graph : langgraph.graph.state.CompiledStateGraph
        The compiled state machine representing the linear conversation flow.
    """
    def __init__(self, nodes):
        """
        Initialize the simple orchestrator and compile its graph.

        Parameters
        ----------
        nodes : Any
            The provider of functional logic for all graph nodes.
        """
        self.nodes = nodes
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Construct and compile a linear LangGraph workflow.

        The flow follows a strict sequence: ingesting the user message, 
        generating a response, and appending the assistant's message 
        to the state.

        Returns
        -------
        langgraph.graph.state.CompiledStateGraph
            A compiled graph instance ready to process `RAGState`.
        """
        builder = StateGraph(RAGState)

        # Register functional nodes
        builder.add_node("add_user_message", self.nodes.add_user_message)
        builder.add_node("add_assistant_message", self.nodes.add_assistant_message)
        builder.add_node("simple_response", self.nodes.simple_response)

        # Define linear execution flow
        builder.add_edge(START, "add_user_message")
        builder.add_edge("add_user_message", "simple_response")
        builder.add_edge("simple_response", "add_assistant_message")
        builder.add_edge("add_assistant_message", END)

        return builder.compile()
