"""
core/risk.py — RiskEngine (weighted scoring)

Risk score composito [0, 1] pesato da RISK_WEIGHT_* in config:
  fairness_component  x W_fairness  (default 0.50)
  drift_component     x W_drift     (default 0.30)
  quality_component   x W_quality   (default 0.20)

Circuit breaker aperto -> forza score=1.0, UNACCEPTABLE.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ethical_governance.config import config
from ethical_governance.infra.metrics import DRIFT_DETECTED
from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.core.fairness import FairnessReport
    from ethical_governance.core.quality import QualityReport
    from ethical_governance.infra.tenancy import CircuitBreaker

logger = get_logger(__name__)


class RiskEngine:
    _MAX_VIOLATIONS = 5
    _SEVERITY_WEIGHTS = {
        "LOW": 0.0, "MEDIUM": 0.25, "HIGH": 0.50, "CRITICAL": 1.0
    }

    def __init__(self, cb: "CircuitBreaker") -> None:
        self._cb = cb

    async def evaluate(
        self,
        tenant_id: str,
        fairness:  Optional["FairnessReport"],
        drift:     Dict[str, Dict[str, Any]],
        quality:   Optional["QualityReport"] = None,
    ) -> Dict[str, Any]:

        if await self._cb.is_open(tenant_id):
            return {
                "risk_score":    1.0,
                "risk_level":    "UNACCEPTABLE",
                "requires_hitl": True,
                "reasons":       ["Circuit breaker aperto."],
                "components":    {"fairness": 1.0, "drift": 1.0, "quality": 1.0},
            }

        reasons: List[str] = []

        fairness_component = 0.0
        if fairness:
            n_viol = len(fairness.violations)
            sev_w  = self._SEVERITY_WEIGHTS.get(fairness.severity, 0.0)
            fairness_component = float(np.clip(
                n_viol / self._MAX_VIOLATIONS + sev_w * config.RISK_WEIGHT_SEVERITY_BOOST,
                0.0, 1.0,
            ))
            if n_viol > 0:
                reasons.append(f"Fairness: {n_viol} violazioni (severity={fairness.severity})")

        drift_component = 0.0
        if drift:
            total     = max(len(drift), 1)
            n_drifted = sum(1 for v in drift.values() if v.get("drift_detected"))
            drift_component = n_drifted / total
            if n_drifted > 0:
                reasons.append(f"Drift: {n_drifted}/{total} feature")
                DRIFT_DETECTED.labels(model_name="—").inc(n_drifted)

        quality_component = 0.0
        if quality and not quality.passed:
            quality_component = float(np.clip(1.0 - quality.quality_score, 0.0, 1.0))
            reasons.append(f"Qualita input: score={quality.quality_score:.2f}")

        score = float(np.clip(
            config.RISK_WEIGHT_FAIRNESS * fairness_component
            + config.RISK_WEIGHT_DRIFT  * drift_component
            + config.RISK_WEIGHT_QUALITY * quality_component,
            0.0, 1.0,
        ))

        level = (
            "UNACCEPTABLE" if score > 0.70 else
            "HIGH"         if score > 0.45 else
            "MEDIUM"       if score > 0.20 else
            "LOW"
        )

        return {
            "risk_score":    round(score, 4),
            "risk_level":    level,
            "requires_hitl": level in config.HITL_REQUIRED_LEVELS,
            "reasons":       reasons,
            "components": {
                "fairness": round(fairness_component, 4),
                "drift":    round(drift_component,    4),
                "quality":  round(quality_component,  4),
            },
            "weights": {
                "fairness": config.RISK_WEIGHT_FAIRNESS,
                "drift":    config.RISK_WEIGHT_DRIFT,
                "quality":  config.RISK_WEIGHT_QUALITY,
            },
        }
