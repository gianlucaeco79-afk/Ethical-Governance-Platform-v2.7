"""
core/drift.py — DriftDetector

KS test per colonne numeriche (int/float/bool).
Chi2 su tabella di contingenza per colonne categoriche (object/category).
Ogni risultato include "detected_type" per trasparenza nel log di audit.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from ethical_governance.infra.observability import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Rileva deriva statistica tra distribuzione di riferimento e serving.

    Rilevamento tipo automatico:
      - dtype.kind in "iufcb" -> numerico -> KS test
      - altrimenti             -> categorico -> Chi2 contingency
    """

    _NUMERIC_KINDS = frozenset("iufcb")

    def __init__(self, p_threshold: float) -> None:
        self.p_threshold = p_threshold

    @classmethod
    def _is_numeric(cls, series: pd.Series) -> bool:
        return series.dtype.kind in cls._NUMERIC_KINDS

    def detect(self, ref: pd.DataFrame, cur: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for col in ref.columns:
            if col not in cur.columns:
                continue
            if self._is_numeric(ref[col]):
                result = self._ks(ref[col], cur[col])
                result["detected_type"] = "numeric"
            else:
                result = self._chi2(ref[col], cur[col])
                result["detected_type"] = "categorical"
            results[col] = result
        return results

    def _ks(self, ref: pd.Series, cur: pd.Series) -> Dict[str, Any]:
        cr = ref.dropna()
        cc = cur.dropna()
        if len(cr) < 2 or len(cc) < 2:
            return {"test": "ks", "statistic": 0.0, "p_value": 1.0,
                    "drift_detected": False, "note": "campione insufficiente"}
        stat, p = stats.ks_2samp(cr, cc)
        return {"test": "ks", "statistic": round(float(stat), 6),
                "p_value": round(float(p), 6), "drift_detected": bool(p < self.p_threshold)}

    def _chi2(self, ref: pd.Series, cur: pd.Series) -> Dict[str, Any]:
        ref_counts = ref.value_counts()
        cur_counts = cur.value_counts()
        all_cats   = sorted(set(ref_counts.index) | set(cur_counts.index))

        if not all_cats:
            return {"test": "chi2", "statistic": 0.0, "p_value": 1.0,
                    "drift_detected": False, "note": "nessuna categoria"}

        observed = np.array([
            [int(ref_counts.get(c, 0)) for c in all_cats],
            [int(cur_counts.get(c, 0)) for c in all_cats],
        ])
        nonzero_cols = observed.sum(axis=0) > 0
        observed     = observed[:, nonzero_cols]

        if observed.shape[1] < 2:
            return {"test": "chi2", "statistic": 0.0, "p_value": 1.0,
                    "drift_detected": False, "note": "categorie insufficienti"}

        try:
            chi2, p, _dof, _exp = stats.chi2_contingency(observed)
        except ValueError:
            return {"test": "chi2", "statistic": 0.0, "p_value": 1.0,
                    "drift_detected": False, "note": "test non applicabile"}

        return {"test": "chi2", "statistic": round(float(chi2), 6),
                "p_value": round(float(p), 6), "drift_detected": bool(p < self.p_threshold)}
