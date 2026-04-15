"""
governance/batch.py — BatchPredictor

Esegue N predizioni in parallelo tramite asyncio.gather.
Partial success: errori per-item non interrompono il batch.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, TYPE_CHECKING

from fastapi import HTTPException

from ethical_governance.infra.metrics import BATCH_REQUESTS, BATCH_SIZE_HIST
from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.governance.engine import GovernanceEngine

logger = get_logger(__name__)


class BatchPredictor:
    def __init__(self, governance: "GovernanceEngine") -> None:
        self._gov = governance

    async def run(
        self,
        tenant_id:  str,
        model_name: str,
        items:      List[Dict[str, float]],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        tasks = [
            self._safe_infer(tenant_id, model_name, features, idx)
            for idx, features in enumerate(items)
        ]
        results = await asyncio.gather(*tasks)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        n_success  = sum(1 for r in results if r.get("status") != "ERROR")
        n_error    = len(results) - n_success

        BATCH_REQUESTS.labels(tenant_id=tenant_id, model_name=model_name).inc()
        BATCH_SIZE_HIST.observe(len(items))

        return {
            "status":     "COMPLETED",
            "model_name": model_name,
            "total":      len(items),
            "success":    n_success,
            "errors":     n_error,
            "latency_ms": latency_ms,
            "results":    list(results),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }

    async def _safe_infer(
        self,
        tenant_id:  str,
        model_name: str,
        features:   Dict[str, float],
        idx:        int,
    ) -> Dict[str, Any]:
        try:
            result = await self._gov.infer(tenant_id, model_name, features)
            result["_batch_index"] = idx
            return result
        except HTTPException as exc:
            return {"status": "ERROR", "_batch_index": idx, "detail": exc.detail}
        except Exception as exc:
            return {"status": "ERROR", "_batch_index": idx, "detail": str(exc)}
