"""
governance/engine.py — GovernanceEngine

Orchestratore del flusso di inferenza completo.
Ogni AUTO_DECISION include campi forensi: request_id, latency_ms, model_version.
AlertService chiamato in background senza bloccare la risposta.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
from fastapi import HTTPException

from ethical_governance.config import config
from ethical_governance.infra.exceptions import FeatureMissingError
from ethical_governance.infra.metrics import PREDICT_LATENCY, PREDICT_REQUESTS
from ethical_governance.infra.observability import get_logger, get_tracer

if TYPE_CHECKING:
    from ethical_governance.core.drift import DriftDetector
    from ethical_governance.core.fairness import FairnessAnalyzer, FairnessReport
    from ethical_governance.core.quality import DataQualityChecker
    from ethical_governance.core.risk import RiskEngine
    from ethical_governance.governance.hitl import HITLManager, DecisionTask
    from ethical_governance.infra.alerts import AlertService
    from ethical_governance.infra.audit import AuditLogger
    from ethical_governance.infra.persistence import ReferenceDataManager, ServingBuffer
    from ethical_governance.infra.tenancy import CircuitBreaker, TenantManager
    from ethical_governance.ml.explainability import ExplainabilityService
    from ethical_governance.ml.models import ModelRegistry
    from ethical_governance.ml.monitor import ModelMonitor
    from ethical_governance.infra.audit import FeedbackStore

logger = get_logger(__name__)


class GovernanceEngine:
    def __init__(
        self,
        registry:        "ModelRegistry",
        ref_mgr:         "ReferenceDataManager",
        audit:           "AuditLogger",
        buffer:          "ServingBuffer",
        fairness:        "FairnessAnalyzer",
        drift:           "DriftDetector",
        quality:         "DataQualityChecker",
        risk:            "RiskEngine",
        hitl:            "HITLManager",
        cb:              "CircuitBreaker",
        monitor:         "ModelMonitor",
        explainability:  "ExplainabilityService",
        feedback_store:  "FeedbackStore",
        alert_service:   "AlertService",
        tenant_manager:  "TenantManager",
    ) -> None:
        self._registry       = registry
        self._ref_mgr        = ref_mgr
        self._audit          = audit
        self._buffer         = buffer
        self._fairness       = fairness
        self._drift          = drift
        self._quality        = quality
        self._risk           = risk
        self._hitl           = hitl
        self._cb             = cb
        self._monitor        = monitor
        self._explainability = explainability
        self._feedback_store = feedback_store
        self._alerts         = alert_service
        self._tm             = tenant_manager

    async def infer(
        self,
        tenant_id:  str,
        model_name: str,
        features:   Dict[str, float],
    ) -> Dict[str, Any]:
        t0         = time.perf_counter()
        request_id = str(uuid.uuid4())

        model   = await self._registry.load(model_name)
        missing = [f for f in model.features if f not in features]
        if missing:
            raise FeatureMissingError(f"Feature mancanti: {missing}")

        input_df = pd.DataFrame([{f: features[f] for f in model.features}])

        ref_meta = self._ref_mgr.get_meta(model.dataset_name)
        quality  = self._quality.check(input_df, ref_meta, model_name)

        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0])

        ref_df         = self._ref_mgr.get_latest(model.dataset_name)
        fairness_limit = self._tm.get_fairness_limit(tenant_id)
        fairness_report: Optional["FairnessReport"] = None
        if ref_df is not None and len(ref_df) >= 50:
            fairness_report = self._fairness.analyze(model, ref_df, fairness_limit)

        await self._buffer.append(
            model.dataset_name,
            {f: features[f] for f in model.features},
        )
        serving_df = await self._buffer.load(model.dataset_name)
        drift_result: Dict[str, Any] = {}
        if ref_df is not None and len(serving_df) >= config.DRIFT_MIN_WINDOW:
            shared = [c for c in model.features if c in serving_df.columns]
            if shared:
                drift_result = self._drift.detect(ref_df[shared], serving_df[shared])

        risk = await self._risk.evaluate(tenant_id, fairness_report, drift_result, quality)
        explanation = self._explainability.explain(model, input_df)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        PREDICT_LATENCY.labels(model_name=model_name).observe(latency_ms / 1000)

        asyncio.create_task(
            self._alerts.evaluate_and_fire(
                model_name, tenant_id, fairness_report,
                drift_result, risk["risk_score"], request_id=request_id,
            ),
            name=f"alert_{request_id}",
        )

        forensic = {
            "request_id":    request_id,
            "latency_ms":    latency_ms,
            "model_version": model.trained_at,
        }

        if risk["requires_hitl"]:
            task_id = str(uuid.uuid4())
            from ethical_governance.governance.hitl import DecisionTask
            task = DecisionTask(
                task_id=task_id, model_key=model_name, input_data=features,
                prediction=pred, probability=prob, risk_report=risk,
                fairness_report=fairness_report.model_dump() if fairness_report else None,
                drift_report=drift_result, quality_report=quality.model_dump(),
                explanation=explanation,
            )
            await self._hitl.store(task)
            await self._audit.log("INTERCEPTION", {
                "tenant_id": tenant_id, "task_id": task_id,
                "model_name": model_name, "risk": risk, **forensic,
            })
            PREDICT_REQUESTS.labels(
                tenant_id=tenant_id, model_name=model_name, status="pending"
            ).inc()
            return {
                "status":          "PENDING_APPROVAL",
                "task_id":         task_id,
                "request_id":      request_id,
                "risk":            risk,
                "quality_report":  quality.model_dump(),
                "fairness_report": fairness_report.model_dump() if fairness_report else None,
                "drift_report":    drift_result,
                "explanation":     explanation,
            }

        prediction_id = str(uuid.uuid4())
        await self._audit.log("AUTO_DECISION", {
            "tenant_id":     tenant_id,
            "model_name":    model_name,
            "prediction_id": prediction_id,
            "prediction":    pred,
            "probability":   prob,
            "risk":          risk,
            **forensic,
        })

        if fairness_report:
            asyncio.create_task(
                self._monitor.record(
                    model_name,
                    accuracy=fairness_report.overall_accuracy,
                    dpd=fairness_report.demographic_parity_difference,
                    request_id=request_id,
                ),
                name=f"monitor_{model_name}",
            )

        PREDICT_REQUESTS.labels(
            tenant_id=tenant_id, model_name=model_name, status="completed"
        ).inc()
        return {
            "status":          "COMPLETED",
            "model_name":      model_name,
            "request_id":      request_id,
            "prediction_id":   prediction_id,
            "prediction":      pred,
            "probability":     round(prob, 6),
            "latency_ms":      latency_ms,
            "model_version":   model.trained_at,
            "risk":            risk,
            "quality_report":  quality.model_dump(),
            "fairness_report": fairness_report.model_dump() if fairness_report else None,
            "drift_report":    drift_result,
            "explanation":     explanation,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }
