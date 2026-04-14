"""
ml/retraining.py — RetrainingPipeline

Chiude il loop feedback -> AutoCorrection -> retraining.
Azione MONITOR -> SKIPPED.
Azione REWEIGH_AND_RETRAIN -> addestra con weight_multiplier suggerito.
Azione RETRAIN_FULL -> addestra da zero con reweighing standard.
Il nuovo modello riceve alias "{base}_retrained_v{n}".
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import joblib
import pandas as pd

from ethical_governance.infra.exceptions import ModelNotFoundError
from ethical_governance.infra.metrics import RETRAIN_TOTAL
from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.infra.audit import AuditLogger
    from ethical_governance.ml.models import ModelRegistry, TrainedModel
    from ethical_governance.governance.correction import AutoCorrectionService

logger = get_logger(__name__)


class RetrainingPipeline:
    def __init__(
        self,
        registry:        "ModelRegistry",
        auto_correction: "AutoCorrectionService",
        audit:           "AuditLogger",
        retrain_dir:     Path,
    ) -> None:
        self._registry  = registry
        self._auto_corr = auto_correction
        self._audit     = audit
        self._dir       = retrain_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _next_version(self, base_name: str) -> str:
        existing = list(self._dir.glob(f"{base_name}_retrained_v*.joblib"))
        if not existing:
            return f"{base_name}_retrained_v1"
        numbers: List[int] = []
        for p in existing:
            try:
                numbers.append(int(p.stem.split("_v")[-1]))
            except ValueError:
                continue
        return (
            f"{base_name}_retrained_v{max(numbers) + 1}"
            if numbers
            else f"{base_name}_retrained_v1"
        )

    async def trigger(
        self,
        base_model_name: str,
        dataset_builder: Callable[[], pd.DataFrame],
        tenant_id:       str,
    ) -> Dict[str, Any]:
        report = await self._auto_corr.generate_report(base_model_name)

        if report.get("status") != "OK":
            return {
                "status":     "SKIPPED",
                "reason":     report.get("message", "INSUFFICIENT_DATA"),
                "model_name": base_model_name,
            }

        action = report.get("suggested_action", "MONITOR")
        if action == "MONITOR":
            return {
                "status":     "SKIPPED",
                "reason":     report["reason"],
                "model_name": base_model_name,
            }

        try:
            base_model = await self._registry.load(base_model_name)
        except ModelNotFoundError as exc:
            return {"status": "ERROR", "reason": str(exc), "model_name": base_model_name}

        weight_multiplier = (
            float(report.get("weight_multiplier", 1.0))
            if action == "REWEIGH_AND_RETRAIN"
            else 1.0
        )
        new_model_name = self._next_version(base_model_name)

        clf_class = base_model.pipeline.named_steps["clf"].__class__.__name__
        algo_map = {
            "LogisticRegression":        "logistic_regression",
            "RandomForestClassifier":    "random_forest",
            "GradientBoostingClassifier": "gradient_boosting",
        }
        algo = algo_map.get(clf_class, "logistic_regression")

        df      = dataset_builder()
        trained = await self._registry.train(
            df,
            model_name        = algo,
            dataset_name      = base_model.dataset_name,
            protected         = base_model.protected,
            target            = base_model.target,
            use_reweighing    = True,
            weight_multiplier = weight_multiplier,
        )

        from ethical_governance.ml.models import TrainedModel
        aliased = TrainedModel(
            name=new_model_name,
            pipeline=trained.pipeline,
            features=trained.features,
            protected=trained.protected,
            target=trained.target,
            dataset_name=trained.dataset_name,
            train_metrics=trained.train_metrics,
            reference_meta=trained.reference_meta,
        )
        joblib.dump(aliased, self._dir / f"{new_model_name}.joblib")
        await self._registry.save(aliased)

        RETRAIN_TOTAL.labels(model_name=base_model_name, action=action).inc()

        await self._audit.log("RETRAIN_TRIGGERED", {
            "tenant_id":         tenant_id,
            "base_model":        base_model_name,
            "new_model":         new_model_name,
            "action":            action,
            "weight_multiplier": weight_multiplier,
            "reason":            report["reason"],
            "new_accuracy":      trained.train_metrics.get("accuracy"),
        })

        return {
            "status":            "COMPLETED",
            "action":            action,
            "base_model":        base_model_name,
            "new_model_name":    new_model_name,
            "weight_multiplier": weight_multiplier,
            "new_accuracy":      trained.train_metrics.get("accuracy"),
            "reason":            report["reason"],
            "triggered_at":      datetime.now(timezone.utc).isoformat(),
        }
