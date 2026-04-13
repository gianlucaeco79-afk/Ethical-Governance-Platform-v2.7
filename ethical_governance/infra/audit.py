"""
infra/audit.py — AuditLogger e FeedbackStore

AuditLogger: log JSONL append-only, thread-safe tramite asyncio.Lock.
FeedbackStore: persistenza ground truth (prediction_id -> real_outcome + tenant_id).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ethical_governance.infra.metrics import FEEDBACK_TOTAL
from ethical_governance.infra.observability import get_logger

logger = get_logger(__name__)


class AuditLoggerBase(ABC):
    """Interfaccia astratta per un audit logger append-only."""

    @abstractmethod
    async def log(self, event_type: str, data: Dict[str, Any]) -> None:
        raise NotImplementedError


class AuditLogger(AuditLoggerBase):
    """
    Log JSONL append-only thread-safe.
    Ogni riga e' un JSON completo con timestamp e tipo evento.
    """

    def __init__(self, path: Path) -> None:
        self.path  = path
        self._lock = asyncio.Lock()

    async def log(self, event_type: str, data: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        async with self._lock:
            await asyncio.to_thread(self._append, entry)

    def _append(self, entry: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def _read_lines(self) -> List[str]:
        return (
            self.path.read_text(encoding="utf-8").splitlines()
            if self.path.exists() else []
        )

    async def read_last(
        self, n: int = 50, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        lines = await asyncio.to_thread(self._read_lines)
        out: List[Dict[str, Any]] = []
        for line in lines:
            try:
                obj = json.loads(line)
                if event_type is None or obj.get("event") == event_type:
                    out.append(obj)
            except json.JSONDecodeError:
                continue
        return out[-n:]

    async def read_all_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        lines = await asyncio.to_thread(self._read_lines)
        out: List[Dict[str, Any]] = []
        for line in lines:
            try:
                obj = json.loads(line)
                if obj.get("event") == event_type:
                    out.append(obj)
            except json.JSONDecodeError:
                continue
        return out

    async def find_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Cerca un singolo evento AUTO_DECISION per prediction_id."""
        events = await self.read_all_by_type("AUTO_DECISION")
        for e in events:
            if e.get("prediction_id") == prediction_id:
                return e
        return None


class FeedbackStore:
    """
    Persiste ground truth reale associato a ogni predizione.
    Il campo tenant_id e' usato per il controllo di ownership.
    """

    def __init__(self, path: Path) -> None:
        self.path  = path
        self._lock = asyncio.Lock()

    async def store(
        self,
        prediction_id: str,
        model_name:    str,
        real_outcome:  int,
        tenant_id:     str,
    ) -> None:
        entry = {
            "prediction_id": prediction_id,
            "model_name":    model_name,
            "real_outcome":  real_outcome,
            "tenant_id":     tenant_id,
            "received_at":   datetime.now(timezone.utc).isoformat(),
        }
        async with self._lock:
            await asyncio.to_thread(self._append, entry)
        FEEDBACK_TOTAL.labels(model_name=model_name).inc()

    def _append(self, entry: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    async def load_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        text = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        out: List[Dict[str, Any]] = []
        for line in text.splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    async def load_for_model(self, model_name: str) -> List[Dict[str, Any]]:
        return [f for f in await self.load_all() if f.get("model_name") == model_name]
