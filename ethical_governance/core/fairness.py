"""
core/fairness.py — FairnessAnalyzer e FairnessReport

Metriche di fairness:
  DPD — Demographic Parity Difference
  DPR — Demographic Parity Ratio (4/5 rule)
  EOD — Equal Opportunity Difference (TPR gap)
  DIR — Disparate Impact Ratio
  PPD — Predictive Parity Difference (Precision gap)

Tutte con bootstrap confidence interval (200 iterazioni).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_score

from ethical_governance.infra.metrics import FAIRNESS_SCORE, FAIRNESS_VIOLATIONS
from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.ml.models import TrainedModel

logger = get_logger(__name__)


class FairnessReport(BaseModel):
    model_key:                     str
    protected_attribute:           str
    group_0_label:                 str
    group_1_label:                 str
    n_group_0:                     int
    n_group_1:                     int
    demographic_parity_difference: float
    demographic_parity_ratio:      float
    equal_opportunity_difference:  float
    disparate_impact_ratio:        float
    predictive_parity_difference:  float
    overall_accuracy:              float
    accuracy_group_0:              float
    accuracy_group_1:              float
    severity:                      str
    violations:                    List[str]
    bootstrap_ci:                  Dict[str, List[float]]
    timestamp:                     str


class FairnessAnalyzer:
    _DIR_THRESHOLD  = 0.80
    _DPR_THRESHOLD  = 0.80
    _MAX_VIOLATIONS = 5

    def analyze(self, model: "TrainedModel", ref_df: pd.DataFrame, fairness_limit: float) -> FairnessReport:
        prot  = model.protected
        label = model.target

        y_true    = ref_df[label].values
        y_pred    = model.predict(ref_df)
        sensitive = ref_df[prot].values

        g0 = sensitive == 0
        g1 = sensitive == 1

        pr0 = float(y_pred[g0].mean()) if g0.sum() > 0 else 0.0
        pr1 = float(y_pred[g1].mean()) if g1.sum() > 0 else 0.0
        dpd = pr0 - pr1
        dpr = (min(pr0, pr1) / max(pr0, pr1)) if max(pr0, pr1) > 0 else 1.0

        pos0 = (y_true == 1) & g0
        pos1 = (y_true == 1) & g1
        tpr0 = float(y_pred[pos0].mean()) if pos0.sum() > 0 else 0.0
        tpr1 = float(y_pred[pos1].mean()) if pos1.sum() > 0 else 0.0
        eod  = tpr0 - tpr1

        dir_r = (min(pr0, pr1) / max(pr0, pr1)) if max(pr0, pr1) > 0 else 1.0

        pp0 = float(precision_score(y_true[g0], y_pred[g0], zero_division=0)) if g0.sum() > 0 else 0.0
        pp1 = float(precision_score(y_true[g1], y_pred[g1], zero_division=0)) if g1.sum() > 0 else 0.0
        ppd = pp0 - pp1

        overall_acc = float(accuracy_score(y_true, y_pred))
        acc0 = float(accuracy_score(y_true[g0], y_pred[g0])) if g0.sum() > 0 else 0.0
        acc1 = float(accuracy_score(y_true[g1], y_pred[g1])) if g1.sum() > 0 else 0.0

        violations: List[str] = []
        for triggered, msg, metric in [
            (abs(dpd)  > fairness_limit,      f"DPD={dpd:.3f} (soglia |{fairness_limit:.2f}|)", "dpd"),
            (dpr       < self._DPR_THRESHOLD, f"DPR={dpr:.3f} (soglia {self._DPR_THRESHOLD})",  "dpr"),
            (abs(eod)  > fairness_limit,      f"EOD={eod:.3f} (soglia |{fairness_limit:.2f}|)", "eod"),
            (dir_r     < self._DIR_THRESHOLD, f"DIR={dir_r:.3f} (soglia {self._DIR_THRESHOLD})","dir"),
            (abs(ppd)  > fairness_limit,      f"PPD={ppd:.3f} (soglia |{fairness_limit:.2f}|)", "ppd"),
        ]:
            if triggered:
                violations.append(msg)
                FAIRNESS_VIOLATIONS.labels(model_name=model.name, metric=metric).inc()

        nv = len(violations)
        severity = (
            "CRITICAL" if nv >= 4 else
            "HIGH"     if nv >= 3 else
            "MEDIUM"   if nv >= 1 else
            "LOW"
        )
        FAIRNESS_SCORE.labels(model_name=model.name).set(abs(dpd))

        return FairnessReport(
            model_key=model.name,
            protected_attribute=prot,
            group_0_label=f"{prot}=0",
            group_1_label=f"{prot}=1",
            n_group_0=int(g0.sum()),
            n_group_1=int(g1.sum()),
            demographic_parity_difference=round(dpd, 6),
            demographic_parity_ratio=round(dpr, 6),
            equal_opportunity_difference=round(eod, 6),
            disparate_impact_ratio=round(dir_r, 6),
            predictive_parity_difference=round(ppd, 6),
            overall_accuracy=round(overall_acc, 6),
            accuracy_group_0=round(acc0, 6),
            accuracy_group_1=round(acc1, 6),
            severity=severity,
            violations=violations,
            bootstrap_ci=self._bootstrap(y_true, y_pred, sensitive),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _bootstrap(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray,
                   n_iter: int = 200, seed: int = 42) -> Dict[str, List[float]]:
        rng  = np.random.default_rng(seed)
        n    = len(y_true)
        vals: List[float] = []
        for _ in range(n_iter):
            idx = rng.choice(n, n, replace=True)
            yp  = y_pred[idx]
            a   = sensitive[idx]
            g0  = yp[a == 0]
            g1  = yp[a == 1]
            p0  = float(g0.mean()) if len(g0) > 0 else 0.0
            p1  = float(g1.mean()) if len(g1) > 0 else 0.0
            vals.append(abs(p0 - p1))
        return {
            "mean_dpd": [round(float(np.mean(vals)), 6)],
            "ci_2_5":   [round(float(np.percentile(vals, 2.5)), 6)],
            "ci_97_5":  [round(float(np.percentile(vals, 97.5)), 6)],
          }
