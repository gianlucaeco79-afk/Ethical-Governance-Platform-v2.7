"""
ml/explainability.py — ExplainabilityService (SHAP + fallback)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.ml.models import TrainedModel

logger = get_logger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore
    SHAP_AVAILABLE = False


class ExplainabilityService:
    """SHAP TreeExplainer/LinearExplainer con doppio fallback."""

    def explain(self, model: "TrainedModel", sample: pd.DataFrame) -> Dict[str, Any]:
        features = model.features
        clf      = model.pipeline.named_steps["clf"]
        scaler   = model.pipeline.named_steps["scaler"]
        X_sc     = scaler.transform(sample[features])

        if SHAP_AVAILABLE and shap is not None:
            result = self._shap(clf, X_sc, features)
            if result:
                return result
        return self._fallback(clf, features)

    @staticmethod
    def _shap(clf: Any, X_sc: np.ndarray, features: List[str]) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(clf, "feature_importances_"):
                exp  = shap.TreeExplainer(clf)
                vals = exp.shap_values(X_sc)
                if isinstance(vals, list):
                    vals = vals[1]
                base = exp.expected_value
                if isinstance(base, np.ndarray):
                    base = base[1] if len(base) > 1 else base[0]
                return {
                    "method": "shap_tree",
                    "feature_contributions": dict(zip(features, [round(float(v), 6) for v in vals[0]])),
                    "base_value": round(float(base), 6),
                }
            if hasattr(clf, "coef_"):
                bg   = X_sc[:min(50, len(X_sc))]
                exp  = shap.LinearExplainer(clf, bg)
                vals = exp.shap_values(X_sc)
                vec  = vals[0] if vals.ndim > 1 else vals
                base = exp.expected_value
                if isinstance(base, np.ndarray):
                    base = base[0]
                return {
                    "method": "shap_linear",
                    "feature_contributions": dict(zip(features, [round(float(v), 6) for v in vec])),
                    "base_value": round(float(base), 6),
                }
        except Exception as exc:
            logger.warning(f"SHAP fallito ({type(exc).__name__}).")
        return None

    @staticmethod
    def _fallback(clf: Any, features: List[str]) -> Dict[str, Any]:
        if hasattr(clf, "feature_importances_"):
            imp    = clf.feature_importances_
            method = "feature_importances"
        elif hasattr(clf, "coef_"):
            imp    = np.abs(clf.coef_[0])
            method = "coef_normalized"
        else:
            imp    = np.ones(len(features))
            method = "uniform_fallback"
        total = imp.sum()
        imp   = imp / total if total > 0 else imp
        return {
            "method": method,
            "feature_contributions": dict(zip(features, [round(float(v), 6) for v in imp])),
            "base_value": None,
        }
