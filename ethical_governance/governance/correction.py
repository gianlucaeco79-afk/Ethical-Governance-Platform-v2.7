"""
governance/correction.py — AutoCorrectionService

Analizza gli override umani per stimare il bias sistematico del modello.
MIN_CORRECTIONS_THRESHOLD applicato per-gruppo: l'azione correttiva viene
suggerita solo se un singolo gruppo accumula >= soglia override omogenei.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Set, TYPE_CHECKING

import numpy as np

from ethical_governance.config import config
from ethical_governance.infra.metrics import AUTO_CORRECTIONS
from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.infra.audit import AuditLogger

logger = get_logger(__name__)


class AutoCorrectionService:
    def __init__(self, audit_logger: "AuditLogger") -> None:
        self._audit = audit_logger

    @property
    def _threshold(self) -> int:
        return config.MIN_CORRECTIONS_THRESHOLD

    async def generate_report(self, model_name: str) -> Dict[str, Any]:
        events    = await self._audit.read_all_by_type("HUMAN_RESOLUTION")
        model_evs = [e for e in events if e.get("model_key") == model_name]
        n_total   = len(model_evs)

        if n_total < config.AUTO_CORRECT_MIN_SAMPLES:
            return {
                "status":    "INSUFFICIENT_DATA",
                "model_name": model_name,
                "count":     n_total,
                "required":  config.AUTO_CORRECT_MIN_SAMPLES,
                "message": (
                    f"Servono almeno {config.AUTO_CORRECT_MIN_SAMPLES} "
                    f"decisioni umane. Disponibili: {n_total}."
                ),
            }

        overrides     = [e for e in model_evs if e.get("resolution") == "overridden"]
        approvals     = [e for e in model_evs if e.get("resolution") == "approved"]
        override_rate = len(overrides) / n_total

        group_stats: Dict[str, Any] = {}
        protected_values: Set[str] = {
            str(e["protected_value"])
            for e in model_evs
            if e.get("protected_value") is not None
        }
        for pv in protected_values:
            grp_evs  = [e for e in model_evs if str(e.get("protected_value")) == pv]
            grp_over = [e for e in grp_evs   if e.get("resolution") == "overridden"]
            group_stats[pv] = {
                "n":               len(grp_evs),
                "overrides":       len(grp_over),
                "override_rate":   round(len(grp_over) / len(grp_evs), 4) if grp_evs else 0.0,
                "above_threshold": len(grp_over) >= self._threshold,
            }

        qualifying = {pv: s for pv, s in group_stats.items() if s["above_threshold"]}
        bias_gap = 0.0
        if len(qualifying) >= 2:
            rates    = [s["override_rate"] for s in qualifying.values()]
            bias_gap = round(max(rates) - min(rates), 4)

        weight_multiplier = round(1.0 + float(np.clip(bias_gap, 0.0, 0.5)), 4)

        if not qualifying:
            action = "MONITOR"
            reason = f"Nessun gruppo ha raggiunto soglia {self._threshold} override. Monitoraggio."
        elif bias_gap > 0.30:
            action = "RETRAIN_FULL"
            reason = f"Bias gap critico ({bias_gap:.2%}). Re-training completo con dataset bilanciato."
        elif override_rate > 0.20:
            action = "REWEIGH_AND_RETRAIN"
            reason = f"Override rate elevato ({override_rate:.2%}). Reweighing x{weight_multiplier:.2f}."
        else:
            action = "MONITOR"
            reason = f"Override rate accettabile ({override_rate:.2%})."

        AUTO_CORRECTIONS.labels(model_name=model_name).inc()

        return {
            "status":                    "OK",
            "model_name":                model_name,
            "total_decisions":           n_total,
            "overrides":                 len(overrides),
            "approvals":                 len(approvals),
            "override_rate":             round(override_rate, 4),
            "bias_gap":                  bias_gap,
            "group_stats":               group_stats,
            "qualifying_groups":         list(qualifying.keys()),
            "min_corrections_threshold": self._threshold,
            "suggested_action":          action,
            "weight_multiplier":         weight_multiplier,
            "reason":                    reason,
            "generated_at":              datetime.now(timezone.utc).isoformat(),
  }
