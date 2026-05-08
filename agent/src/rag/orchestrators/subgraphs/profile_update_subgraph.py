from langgraph.graph import END, START, StateGraph

from src.rag.schemas.graph_state import RAGState

from .base import BaseSubgraph


class ProfileUpdateSubgraph(BaseSubgraph):
    """
    Subgraph responsible for updating the user profile or context within the RAG pipeline.
    """
    def __init__(self, nodes):
        """
        Initialize the ProfileUpdateSubgraph.

        Parameters
        ----------
        nodes : NodeRegistry
            A registry or container providing access to the functional nodes, 
            specifically the 'update_profile' node.
        """
        super().__init__(nodes)
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Construct and compile the state graph for profile updates.

        Defines a simple linear flow: START -> update_profile -> END.

        Returns
        -------
        CompiledGraph
            The compiled state graph ready to be invoked as a node in the main graph.
        """
        # Initialize the graph with the shared RAGState schema
        builder = StateGraph(RAGState)

        # Register the profile update node from the nodes registry
        builder.add_node("update_profile", self.nodes.update_profile)

        # Define the workflow edges
        builder.add_edge(START, "update_profile")
        builder.add_edge("update_profile", END)

        return builder.compile()
