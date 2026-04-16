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
from src.rag.vectorstore.factory import VectorStoreFactory
from src.rag.retriever.factory import RetrieverFactory

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

    # Final application context
    return AppContext(
        settings=config,
        embeddings=embeddings,
        vectorstore=vectorstore,
    )
