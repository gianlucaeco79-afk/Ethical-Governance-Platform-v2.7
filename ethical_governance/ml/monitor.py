"""
ml/monitor.py — ModelMonitor

Traccia accuracy e DPD nel tempo per ogni modello.
compute_real_accuracy() incrocia predizioni con feedback ground truth.
La scrittura dei log include request_id quando disponibile, per tracing end-to-end.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ethical_governance.infra.metrics import MODEL_DRIFT_GAUGE, REAL_ACCURACY_GAUGE
from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.infra.audit import AuditLogger, FeedbackStore

logger = get_logger(__name__)


class ModelMonitor:
    _DEGRADATION_THRESHOLD = 0.05

    def __init__(self, monitor_dir: Path) -> None:
        self._dir  = monitor_dir
        self._lock = asyncio.Lock()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name: str) -> Path:
        return self._dir / f"{model_name}_monitor.jsonl"

    async def record(
        self, model_name: str, accuracy: float, dpd: float,
        request_id: Optional[str] = None
    ) -> None:
        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "accuracy":   round(accuracy, 6),
            "dpd":        round(abs(dpd), 6),
            "request_id": request_id,
        }
        async with self._lock:
            await asyncio.to_thread(self._append, model_name, entry)

    def _append(self, model_name: str, entry: Dict[str, Any]) -> None:
        with self._path(model_name).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    async def get_history(self, model_name: str, last_n: int = 100) -> List[Dict[str, Any]]:
        path = self._path(model_name)
        if not path.exists():
            return []
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        out: List[Dict[str, Any]] = []
        for line in text.splitlines()[-last_n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    async def degradation_report(self, model_name: str, baseline_accuracy: float) -> Dict[str, Any]:
        history = await self.get_history(model_name, last_n=20)
        if len(history) < 5:
            return {"status": "INSUFFICIENT_DATA", "count": len(history)}
        recent_acc = float(np.mean([h["accuracy"] for h in history]))
        delta      = baseline_accuracy - recent_acc
        MODEL_DRIFT_GAUGE.labels(model_name=model_name).set(delta)
        return {
            "model_name":        model_name,
            "baseline_accuracy": round(baseline_accuracy, 6),
            "recent_accuracy":   round(recent_acc, 6),
            "accuracy_delta":    round(delta, 6),
            "degraded":          delta > self._DEGRADATION_THRESHOLD,
            "alert":             delta > self._DEGRADATION_THRESHOLD,
            "snapshots_used":    len(history),
        }

    async def compute_real_accuracy(
        self, model_name: str, audit: "AuditLogger", feedback_store: "FeedbackStore"
    ) -> Dict[str, Any]:
        auto_decisions = await audit.read_all_by_type("AUTO_DECISION")
        predictions: Dict[str, int] = {
            e["prediction_id"]: e["prediction"]
            for e in auto_decisions
            if e.get("model_name") == model_name and "prediction_id" in e
        }

        feedbacks = await feedback_store.load_for_model(model_name)
        if not feedbacks:
            return {
                "status":     "NO_FEEDBACK",
                "model_name": model_name,
                "n_feedback": 0,
                "message":    "Nessun feedback ricevuto. Usa POST /v1/feedback.",
            }

        matched: List[Dict[str, int]] = []
        unmatched_ids: List[str]      = []
        for fb in feedbacks:
            pid = fb.get("prediction_id", "")
            if pid in predictions:
                matched.append({"predicted": predictions[pid], "real": fb["real_outcome"]})
            else:
                unmatched_ids.append(pid)

        if len(matched) < 5:
            return {
                "status":      "INSUFFICIENT_MATCHED",
                "model_name":  model_name,
                "n_feedback":  len(feedbacks),
                "n_matched":   len(matched),
                "n_unmatched": len(unmatched_ids),
                "message":     f"Servono almeno 5 feedback matchati. Disponibili: {len(matched)}.",
            }

        y_pred    = np.array([m["predicted"] for m in matched])
        y_true    = np.array([m["real"]      for m in matched])
        real_acc  = float(accuracy_score(y_true, y_pred))
        real_prec = float(precision_score(y_true, y_pred, zero_division=0))
        real_rec  = float(recall_score(y_true, y_pred, zero_division=0))

        REAL_ACCURACY_GAUGE.labels(model_name=model_name).set(real_acc)

        return {
            "status":         "OK",
            "model_name":     model_name,
            "n_feedback":     len(feedbacks),
            "n_matched":      len(matched),
            "n_unmatched":    len(unmatched_ids),
            "real_accuracy":  round(real_acc, 6),
            "real_precision": round(real_prec, 6),
            "real_recall":    round(real_rec, 6),
            "computed_at":    datetime.now(timezone.utc).isoformat(),
        }
