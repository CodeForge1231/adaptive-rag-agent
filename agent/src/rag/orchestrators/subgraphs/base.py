from abc import ABC, abstractmethod


class BaseSubgraph(ABC):
    def __init__(self, nodes):
        """
        Abstract base class for defining LangGraph subgraphs.

        Attributes
        ----------
        nodes : Any
            A collection of callable components or logic providers used as 
            nodes within the graph.
        graph : langgraph.graph.state.CompiledStateGraph
            The compiled executable graph built during initialization.
        """
        self.nodes = nodes
        self.graph = self._build_graph()

    @abstractmethod
    def _build_graph(self):
        pass
