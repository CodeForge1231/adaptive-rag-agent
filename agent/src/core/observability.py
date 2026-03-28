import functools
import logging
import time
from typing import Any, Callable, Optional

# OpenTelemetry Core
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Exporters (gRPC)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

# Instrumentations
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# SDK & Resources
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, sampling

# Processors & Readers
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from .config import LangchainConfig, ObservabilityConfig, OtelConfig, ServiceConfig


class Observability:
    """
    A Singleton class to manage OpenTelemetry life-cycle including Traces,
    Metrics, and Logs.

    This class initializes the OTLP exporters, configures global providers,
    and provides decorators for easy instrumentation of app components.
    """

    _instance = None

    def __new__(cls):
        """
        Ensure only one instance of Observability exists (Singleton pattern).
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def init(self, cfg: ObservabilityConfig, orchestrator_strategy: str) -> None:
        """
        Initialize OpenTelemetry providers (traces, metrics, logs).

        Parameters
        ----------
        cfg : ObservabilityConfig
            Configuration settings for observability signals and exporters.
        orchestrator_strategy : str
            The name of the RAG strategy being used, for resource attribution.
        """

        # Prevent double-initialization
        if self._initialized:
            return

        if not cfg.enabled:
            return

        # Store orchestrator strategy for span-level attribution
        self._orchestrator_strategy = orchestrator_strategy

        # Resolve sub-configs with safe defaults
        service_cfg = cfg.service or ServiceConfig(name="unknown")
        otel_cfg = cfg.otel
        lc_cfg = cfg.instrument_langchain or LangchainConfig()

        # Define service identity once and reuse across all OTEL signals
        resource = Resource.create(
            {
                "service.name": service_cfg.name,
                "deployment.environment": service_cfg.environment,
                "rag.orchestrator": orchestrator_strategy,
            }
        )

        # Traces / Metrics / Logs
        if otel_cfg:
            self._setup_tracing(resource, otel_cfg)
            self._setup_metrics(resource, otel_cfg)
            self._setup_logging(resource, otel_cfg)

        # LangChain instrumentation
        if lc_cfg.enabled:
            self._setup_langchain(capture_content=lc_cfg.capture_content)

        self._initialized = True
        logging.info("Observability initialized")

    def _setup_tracing(self, resource: Resource, otel_cfg: OtelConfig) -> None:
        """
        Configure the Trace Provider and Batch Span Processing.

        Parameters
        ----------
        resource : Resource
            The OpenTelemetry resource representing the service.
        otel_cfg : OtelConfig
            Tracing parameters like sample rate and batch sizes.
        """
        sampler = sampling.TraceIdRatioBased(otel_cfg.sample_rate)
        provider = TracerProvider(resource=resource, sampler=sampler)

        exporter = OTLPSpanExporter(endpoint=otel_cfg.endpoint, insecure=True)

        # BatchSpanProcessor settings extracted to config
        processor = BatchSpanProcessor(
            exporter,
            schedule_delay_millis=otel_cfg.trace_batch_delay,
            max_export_batch_size=otel_cfg.trace_batch_size,
            export_timeout_millis=otel_cfg.trace_export_timeout,
        )

        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(resource.attributes["service.name"])

    def _setup_metrics(self, resource: Resource, otel_cfg: OtelConfig) -> None:
        """
        Configure the Metric Provider and Periodic Reading.

        Parameters
        ----------
        resource : Resource
            The OpenTelemetry resource representing the service.
        otel_cfg : OtelConfig
            Metric settings like export intervals.
        """
        exporter = OTLPMetricExporter(endpoint=otel_cfg.endpoint, insecure=True)

        reader = PeriodicExportingMetricReader(
            exporter, export_interval_millis=otel_cfg.metrics_export_interval
        )

        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)

        self._meter = metrics.get_meter(resource.attributes["service.name"])

        # Standard Instruments
        self._requests_total = self._meter.create_counter(
            name="rag_requests_total", description="Total number of RAG pipeline executions"
        )
        self._latency = self._meter.create_histogram(
            name="rag_stage_latency_ms",
            unit="ms",
            description="Latency of specific RAG pipeline stages",
        )

    def _setup_logging(self, resource: Resource, otel_cfg: OtelConfig) -> None:
        """
        Configure OTLP Log Record Processing and Root Logger Integration.

        Parameters
        ----------
        resource : Resource
            The OpenTelemetry resource representing the service.
        otel_cfg : OtelConfig
            Log batching settings.
        """
        # Trace Context injection in logs
        LoggingInstrumentor().instrument(set_logging_format=True)

        provider = LoggerProvider(resource=resource)
        exporter = OTLPLogExporter(endpoint=otel_cfg.endpoint, insecure=True)

        provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter, schedule_delay_millis=otel_cfg.logs_batch_delay)
        )
        set_logger_provider(provider)

        # Attach OTEL handler to Python standard logging
        handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    def _setup_langchain(self, capture_content: bool) -> None:
        """
        Enable automated LangChain tracing if not already active.

        Parameters
        ----------
        capture_content : bool
            Whether to include raw prompts and responses in traces.
        """
        if not LangchainInstrumentor().is_instrumented_by_opentelemetry:
            LangchainInstrumentor().instrument(capture_content=capture_content)

    def _start_span(self, name: str, func: Callable) -> Any:
        """
        Internal helper to create a span with consistent metadata.

        Parameters
        ----------
        name : str
            Name of the span.
        func : Callable
            The function being instrumented.

        Returns
        -------
        Span
            The active span context.
        """
        return self._tracer.start_as_current_span(
            name,
            attributes={
                "code.function": func.__name__,
                "code.namespace": func.__module__,
                "rag.orchestrator": getattr(self, "_orchestrator_strategy", "unknown"),
            },
        )

    def time_it_async(self, name: Optional[str] = None):
        """
        Decorator for asynchronous functions to measure latency and record traces.

        Parameters
        ----------
        name : str, optional
            Custom name for the Span. Defaults to the function name.
        """

        def decorator(func):
            span_name = name or func.__name__

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if not getattr(self, "_tracer", None):
                    return await func(*args, **kwargs)

                start_time = time.perf_counter()
                with self._start_span(span_name, func) as span:
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        if hasattr(self, "_latency"):
                            self._latency.record(duration_ms, attributes={"span": span_name})

            return wrapper

        return decorator

    def time_it(self, name: Optional[str] = None):
        """
        Decorator for synchronous functions to measure latency and record traces.

        Parameters
        ----------
        name : str, optional
            Custom name for the Span. Defaults to the function name.
        """

        def decorator(func):
            span_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not getattr(self, "_tracer", None):
                    return func(*args, **kwargs)

                start_time = time.perf_counter()
                with self._start_span(span_name, func) as span:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                    finally:
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        if hasattr(self, "_latency"):
                            self._latency.record(duration_ms, attributes={"span": span_name})

            return wrapper

        return decorator


# Instance
observability = Observability()
