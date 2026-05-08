from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from src.core.observability import observability


class OrchestratorRequirement(Enum):
    """
    Enumeration of functional requirements for an orchestrator.
    """
    PERSISTENCE = auto()


class BaseOrchestrator(ABC):
    """
    Abstract base class for high-level graph orchestrators.

    This class provides a standardized interface for building, visualizing, 
    and executing complex LangGraph workflows. It includes built-in 
    observability and requirement tracking.

    Attributes
    ----------
    REQUIRED : set[OrchestratorRequirement]
        A set of requirements that the orchestrator must satisfy.
    nodes : Any
        Implementation logic for the graph nodes.
    deps : dict
        Additional dependencies required for graph execution or initialization.
    graph : CompiledStateGraph
        The compiled, executable state machine instance.
    """
    REQUIRED: set[OrchestratorRequirement] = set()

    def __init__(self, nodes, **deps):
        """
        Initialize the orchestrator and compile the underlying graph.

        Parameters
        ----------
        nodes : Any
            The provider of node functions.
        **deps : dict
            Arbitrary keyword arguments representing external dependencies 
            (e.g., database clients, API wrappers).
        """
        self.nodes = nodes
        self.deps = deps
        self.graph: CompiledStateGraph = self._build_graph()

    @abstractmethod
    def _build_graph(self) -> CompiledStateGraph:
        """
        Construct the graph topology and compile it.

        Must be implemented by subclasses to define specific RAG or 
        agentic workflows.

        Returns
        -------
        CompiledStateGraph
            The finalized LangGraph instance ready for invocation.
        """
        pass

    def save_visualization(self, output_path):
        """
        Generate and save a Mermaid diagram of the graph structure.

        Parameters
        ----------
        output_path : str or Path
            The file system path where the PNG visualization will be stored.
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image_data = self.graph.get_graph(xray=True).draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(image_data)
        except Exception:
            pass

    @classmethod
    def get_requirements(cls) -> set[OrchestratorRequirement]:
        """
        Retrieve the set of requirements defined for this orchestrator.

        Returns
        -------
        set[OrchestratorRequirement]
            The collection of mandatory functional requirements.
        """
        return cls.REQUIRED

    @observability.time_it_async("Full Pipeline Execution")
    async def run(self, state: dict):
        """
        Execute the compiled graph asynchronously.

        Parameters
        ----------
        state : dict
            The initial input state for the graph execution.

        Returns
        -------
        dict
            The final state after the graph completes all node transitions.
        """
        return await self.graph.ainvoke(state)
