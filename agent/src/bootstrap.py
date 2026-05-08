from src.core.config import (
    LangchainConfig,
    LoggingConfig,
    ObservabilityConfig,
    OtelConfig,
    ServiceConfig,
    TracebackConfig,
)
from src.core.load_config import load_config
from src.core.logging import setup_logging
from src.core.observability import observability
from src.core.traceback import setup_traceback
from src.rag.embeddings.factory import EmbeddingFactory
from src.rag.llm.factory import LLMFactory
from src.rag.nodes import RAGNodes
from src.rag.orchestrators.base import OrchestratorRequirement
from src.rag.orchestrators.factory import OrchestratorFactory
from src.rag.vectorstore.factory import VectorStoreFactory
from src.rag.retriever.factory import RetrieverFactory
from src.rag.persistence.factory import PersistenceFactory
from src.rag.reranker.factory import RerankerFactory
from src.rag.chains_library import ChainLibrary


from .app_context import AppContext


async def bootstrap():
    """
    Application bootstrap function.
    """

    # Core configuration & infrastructure
    config = load_config()

    setup_traceback(TracebackConfig(**config.get("app").get("traceback")))
    setup_logging(LoggingConfig(**config.get("app").get("logging")))

    obs_cfg = config.get("app").get("observability", {})
    observability.init(
        ObservabilityConfig(
            enabled=obs_cfg.get("enabled", True),
            service=ServiceConfig(**obs_cfg["service"]) if "service" in obs_cfg else None,
            otel=OtelConfig(**obs_cfg["otel"]) if "otel" in obs_cfg else None,
            instrument_langchain=(
                LangchainConfig(**obs_cfg["instrument_langchain"])
                if "instrument_langchain" in obs_cfg
                else None
            ),
        ),
        orchestrator_strategy=(
            config.get("app").get("rag", {}).get("orchestrator", {}).get("strategy", "default")
        ),
    )

    # Core RAG components
    embeddings = EmbeddingFactory.create(config["app"]["rag"]["embeddings"])
    vectorstore = VectorStoreFactory.create(config["app"]["rag"]["vectorstore"], embeddings)
    retriever = RetrieverFactory.create(
        raw_cfg=config["app"]["rag"]["retriever"],
        vector_store=vectorstore,
        vectorstore_provider=config["app"]["rag"]["vectorstore"]["provider"],
    )

    # LLMs with different performance
    llm_fast = LLMFactory.create(config["app"]["rag"]["llm"]["fast"])

    llm_heavy = LLMFactory.create(config["app"]["rag"]["llm"]["heavy"])

    # Library of reusable LLM chains
    chains = ChainLibrary(fast_llm=llm_fast, heavy_llm=llm_heavy)

    # Optional reranking layer for retrieved documents
    reranker = RerankerFactory.create(config["app"]["rag"]["reranker"], chains)

    # Orchestrator selection
    orchestrator_strategy = config["app"]["rag"]["orchestrator"]["strategy"]
    requirements = OrchestratorFactory.get_requirements(orchestrator_strategy)

    # Persistence-related dependencies
    profile_repo = None
    document_history_repo = None
    persistence = None

    # Initialize persistence only if required by the orchestrator
    if OrchestratorRequirement.PERSISTENCE in requirements:
        persistence = await PersistenceFactory.create(config["app"]["rag"]["persistence"])

        profile_repo = persistence.repositories["user_profiles"]
        document_history_repo = persistence.repositories["document_history"]

    # RAG execution nodes
    nodes = RAGNodes(
        config=config,
        chains=chains,
        retriever=retriever,
        reranker=reranker,
        profile_repo=profile_repo,
        document_history_repo=document_history_repo,
    )

    # Build orchestrator based on selected strategy
    orchestrator = OrchestratorFactory.create(
        orchestrator_strategy,
        nodes,
    )

    # Optional graph visualization export
    graph_viz_cfg = config["app"]["output"]["graph_viz"]
    if graph_viz_cfg.get("auto_save"):
        orchestrator.save_visualization(graph_viz_cfg["file_path"])

    # Final application context
    return AppContext(
        settings=config,
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        persistence=persistence,
        user_profiles=profile_repo,
        document_history=document_history_repo,
        orchestrator=orchestrator,
    )
