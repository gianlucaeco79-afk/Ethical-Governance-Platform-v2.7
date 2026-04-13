"""
infra/observability.py — ObservabilityService (pattern da V15.5)

Fornisce un punto unico per logger strutturato e tracer.
Il tracer è MockTracer in sviluppo; sostituibile con OpenTelemetry
senza toccare nessun altro modulo.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class Span:
    """Span no-op. In produzione: opentelemetry.trace.Span."""
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


class TracerBase:
    """Interfaccia minima compatibile con OpenTelemetry Tracer."""
    def start_as_current_span(self, name: str) -> Span:
        raise NotImplementedError


class MockTracer(TracerBase):
    """Tracer no-op per sviluppo e test."""
    def start_as_current_span(self, name: str) -> Span:
        return Span()


class OTelTracer(TracerBase):
    """
    Wrapper per OpenTelemetry reale.
    In produzione:
        observability.set_tracer(OTelTracer(trace.get_tracer("ethical_governance")))
    """
    def __init__(self, otel_tracer: Any) -> None:
        self._tracer = otel_tracer

    def start_as_current_span(self, name: str) -> Any:
        return self._tracer.start_as_current_span(name)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level":   record.levelname,
            "message": record.getMessage(),
            "logger":  record.name,
            "extra":   getattr(record, "extra", {}),
        }, default=str)


class ObservabilityService:
    """
    Punto unico per logging strutturato e tracing.
    Tracer: MockTracer di default; sostituibile via set_tracer()
    senza toccare nessun altro modulo.
    """

    def __init__(self) -> None:
        self._tracer: TracerBase = MockTracer()
        self._loggers: dict[str, logging.Logger] = {}

    def get_logger(self, name: str) -> logging.Logger:
        if name in self._loggers:
            return self._loggers[name]
        log = logging.getLogger(name)
        if not log.handlers:
            h = logging.StreamHandler()
            h.setFormatter(JsonFormatter())
            log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
        self._loggers[name] = log
        return log

    def get_tracer(self) -> TracerBase:
        return self._tracer

    def set_tracer(self, tracer: TracerBase) -> None:
        """Sostituisce il tracer a runtime (es. OpenTelemetry in produzione)."""
        self._tracer = tracer


observability = ObservabilityService()


def get_logger(name: str) -> logging.Logger:
    return observability.get_logger(name)


def get_tracer() -> TracerBase:
    return observability.get_tracer()
