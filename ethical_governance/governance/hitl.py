"""
governance/hitl.py — HITLManager e DecisionTask

Gestisce la coda dei task in attesa di revisione umana.
Redis con TTL 24h o fallback su file JSON per task.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ethical_governance.infra.exceptions import TaskNotFoundError
from ethical_governance.infra.metrics import HITL_PENDING, HITL_RESOLVED
from ethical_governance.infra.observability import get_logger

logger = get_logger(__name__)


class DecisionTask(BaseModel):
    task_id:         str
    model_key:       str
    input_data:      Dict[str, Any]
    prediction:      int
    probability:     float
    risk_report:     Dict[str, Any]
    fairness_report: Optional[Dict[str, Any]] = None
    drift_report:    Optional[Dict[str, Any]] = None
    quality_report:  Optional[Dict[str, Any]] = None
    explanation:     Optional[Dict[str, Any]] = None
    status:          str = "PENDING"
    created_at:      str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at:     Optional[str] = None
    reviewer_note:   Optional[str] = None
    final_decision:  Optional[int] = None
    resolution:      Optional[str] = None


class HITLManager:
    def __init__(self, pending_dir: Path, redis_client: Optional[Any] = None) -> None:
        self.pending_dir = pending_dir
        self._redis      = redis_client
        self._lock       = asyncio.Lock()

    def _path(self, task_id: str) -> Path:
        return self.pending_dir / f"{task_id}.json"

    async def store(self, task: DecisionTask) -> None:
        payload = task.model_dump_json(indent=2)
        async with self._lock:
            if self._redis:
                await self._redis.setex(f"hitl:{task.task_id}", 86400, payload)
            else:
                await asyncio.to_thread(self._path(task.task_id).write_text, payload, "utf-8")
        HITL_PENDING.inc()

    async def get(self, task_id: str) -> Optional[DecisionTask]:
        if self._redis:
            raw = await self._redis.get(f"hitl:{task_id}")
            return DecisionTask.model_validate_json(raw) if raw else None
        path = self._path(task_id)
        if not path.exists():
            return None
        text = await asyncio.to_thread(path.read_text, "utf-8")
        return DecisionTask.model_validate_json(text)

    async def list_pending(self) -> List[DecisionTask]:
        if self._redis:
            keys = await self._redis.keys("hitl:*")
            out: List[DecisionTask] = []
            for k in keys:
                raw = await self._redis.get(k)
                if raw:
                    try:
                        out.append(DecisionTask.model_validate_json(raw))
                    except Exception:
                        continue
            return sorted(out, key=lambda t: t.created_at, reverse=True)

        results: List[DecisionTask] = []
        for p in self.pending_dir.glob("*.json"):
            try:
                text = await asyncio.to_thread(p.read_text, "utf-8")
                results.append(DecisionTask.model_validate_json(text))
            except Exception:
                continue
        return sorted(results, key=lambda t: t.created_at, reverse=True)

    async def resolve(
        self,
        task_id:        str,
        approved:       bool,
        reviewer_note:  Optional[str] = None,
        override_value: Optional[int] = None,
    ) -> DecisionTask:
        task = await self.get(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task '{task_id}' non trovato.")

        task.status         = "APPROVED" if approved else "OVERRIDDEN"
        task.resolved_at    = datetime.now(timezone.utc).isoformat()
        task.reviewer_note  = reviewer_note
        task.final_decision = task.prediction if override_value is None else override_value
        task.resolution     = "approved" if approved else "overridden"

        await self.store(task)
        if not self._redis:
            await asyncio.to_thread(self._path(task_id).unlink, missing_ok=True)

        HITL_PENDING.dec()
        HITL_RESOLVED.labels(resolution=task.resolution).inc()
        return task
