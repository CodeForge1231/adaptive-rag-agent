from .simple_orchestrator import SimpleOrchestrator


class OrchestratorFactory:
    """
    A factory class for managing and instantiating RAG orchestrators.

    Attributes
    ----------
    REGISTRY : dict[str, Type[BaseOrchestrator]]
        A mapping of strategy names to their corresponding orchestrator classes.
    """

    REGISTRY = {
        "simple_orchestrator": SimpleOrchestrator,
    }

    @classmethod
    def get_requirements(cls, strategy: str) -> set:
        """
        Retrieve the functional requirements for a specific orchestrator strategy.

        This method allows checking for necessary dependencies (like persistence) 
        without the overhead of instantiating the orchestrator class.

        Parameters
        ----------
        strategy : str
            The key identifying the desired orchestrator in the REGISTRY.

        Returns
        -------
        set
            A set of requirements associated with the selected orchestrator class.

        Raises
        ------
        ValueError
            If the provided strategy key is not found in the REGISTRY.
        """

        try:
            orch_cls = cls.REGISTRY[strategy]
        except KeyError:
            raise ValueError(f"Unknown orchestrator strategy: {strategy}")

        return orch_cls.get_requirements()

    @classmethod
    def create(cls, strategy: str, nodes):
        """
        Instantiate the orchestrator corresponding to the selected strategy.

        Parameters
        ----------
        strategy : str
            The key identifying the desired orchestrator in the REGISTRY.
        nodes : Any
            The node implementations required by the orchestrator.

        Returns
        -------
        BaseOrchestrator
            An initialized instance of the requested orchestrator.

        Raises
        ------
        ValueError
            If the provided strategy key is not found in the REGISTRY.
        """

        try:
            orch_cls = cls.REGISTRY[strategy]
        except KeyError:
            raise ValueError(f"Unknown orchestrator strategy: {strategy}")

        return orch_cls(nodes)
