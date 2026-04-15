"""
governance/comparison.py — ModelComparator

Confronta fairness e accuracy di piu modelli sullo stesso campione.
Produce ranking e suggerisce quale modello promuovere in produzione.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, TYPE_CHECKING

from ethical_governance.infra.exceptions import ModelNotFoundError
from ethical_governance.infra.metrics import COMPARE_REQUESTS
from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.core.fairness import FairnessAnalyzer
    from ethical_governance.infra.persistence import ReferenceDataManager
    from ethical_governance.infra.tenancy import TenantManager
    from ethical_governance.ml.models import ModelRegistry

logger = get_logger(__name__)


class ModelComparator:
    def __init__(
        self,
        registry:       "ModelRegistry",
        ref_manager:    "ReferenceDataManager",
        fairness:       "FairnessAnalyzer",
        tenant_manager: "TenantManager",
    ) -> None:
        self._registry = registry
        self._ref_mgr  = ref_manager
        self._fairness = fairness
        self._tm       = tenant_manager

    async def compare(
        self,
        model_names: List[str],
        tenant_id:   str,
        sample_size: int = 500,
    ) -> Dict[str, Any]:
        if len(model_names) < 2:
            raise ValueError("Confronto richiede almeno 2 modelli.")
        if len(model_names) > 10:
            raise ValueError("Massimo 10 modelli per confronto.")

        fairness_limit = self._tm.get_fairness_limit(tenant_id)
        reports: Dict[str, Any] = {}
        errors:  Dict[str, str] = {}

        for name in model_names:
            try:
                model  = await self._registry.load(name)
                ref_df = self._ref_mgr.get_latest(model.dataset_name)
                if ref_df is None or len(ref_df) < 50:
                    errors[name] = "Dataset di riferimento insufficiente."
                    continue
                sample        = ref_df.sample(n=min(sample_size, len(ref_df)), random_state=42)
                report        = self._fairness.analyze(model, sample, fairness_limit)
                reports[name] = report.model_dump()
            except ModelNotFoundError as exc:
                errors[name] = str(exc)
            except Exception as exc:
                errors[name] = f"Errore: {exc}"

        if not reports:
            return {"status": "ERROR", "errors": errors, "models": model_names}

        _SEV_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

        ranking_accuracy = sorted(
            reports.keys(),
            key=lambda n: reports[n]["overall_accuracy"],
            reverse=True,
        )
        ranking_fairness = sorted(
            reports.keys(),
            key=lambda n: (
                _SEV_ORDER.get(reports[n]["severity"], 4),
                abs(reports[n]["demographic_parity_difference"]),
            ),
        )

        candidates = [
            n for n in reports
            if _SEV_ORDER.get(reports[n]["severity"], 4) <= 1
        ]
        if candidates:
            best   = max(candidates, key=lambda n: reports[n]["overall_accuracy"])
            reason = (
                f"'{best}' ha la migliore accuracy "
                f"({reports[best]['overall_accuracy']:.4f}) "
                f"tra i modelli con severity <= MEDIUM."
            )
        else:
            best   = min(reports.keys(), key=lambda n: len(reports[n]["violations"]))
            reason = (
                f"'{best}' ha il minor numero di violazioni fairness "
                f"({len(reports[best]['violations'])})."
            )

        COMPARE_REQUESTS.labels(tenant_id=tenant_id).inc()

        return {
            "status":                "OK",
            "models_compared":       list(reports.keys()),
            "errors":                errors,
            "fairness_reports":      reports,
            "ranking_accuracy":      ranking_accuracy,
            "ranking_fairness":      ranking_fairness,
            "recommended_model":     best,
            "recommendation_reason": reason,
            "sample_size":           sample_size,
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }
