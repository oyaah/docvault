"""Observability setup — structured logging (structlog) + distributed tracing (OpenTelemetry).

structlog: JSON-formatted logs, searchable in CloudWatch/Loki/DataDog.
OpenTelemetry: Request-level spans with trace propagation across API → Celery → Redis.
"""

import logging

import structlog
from docvault.config import settings


def configure_logging():
    """Configure structlog for JSON output with stdlib integration."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog
    logging.basicConfig(format="%(message)s", level=logging.INFO, force=True)


def configure_telemetry(app=None):
    """Configure OpenTelemetry tracing with optional OTLP export."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name": "docvault",
            "service.version": "0.2.0",
        })
        provider = TracerProvider(resource=resource)

        if settings.otel_exporter_endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        # Auto-instrument FastAPI
        if app:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)

        structlog.get_logger().info("opentelemetry_configured",
                                    endpoint=settings.otel_exporter_endpoint or "stdout")
    except Exception as e:
        structlog.get_logger().warning("opentelemetry_setup_failed", error=str(e))
