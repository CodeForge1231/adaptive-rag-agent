from __future__ import annotations

from dataclasses import dataclass


#  Configuration for application logging.
@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True
    level: str = "info"
    rich_tracebacks: bool = True
    show_time: bool = True
    show_level: bool = True
    show_path: bool = False


# Configuration for exception tracebacks.
@dataclass(frozen=True)
class TracebackConfig:
    enabled: bool = True
    show_locals: dict[str, bool] | None = None
    suppress: list[str] | None = None


# Configuration for observability features such as tracing, metrics, and logs.
@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = True
    service: ServiceConfig | None = None
    otel: OtelConfig | None = None
    instrument_langchain: LangchainConfig | None = None


# Service identification used in observability systems.
@dataclass(frozen=True)
class ServiceConfig:
    name: str
    environment: str = "dev"


# OpenTelemetry exporter configuration.
@dataclass(frozen=True)
class OtelConfig:
    endpoint: str

    # Sampling
    sample_rate: float = 1.0

    # Traces
    trace_batch_delay: int = 1000
    trace_batch_size: int = 512
    trace_export_timeout: int = 5000

    # Metrics
    metrics_export_interval: int = 5000

    # Logs
    logs_batch_delay: int = 1000


# Configuration for LangChain instrumentation.
@dataclass(frozen=True)
class LangchainConfig:
    enabled: bool = False
    capture_content: bool = False
