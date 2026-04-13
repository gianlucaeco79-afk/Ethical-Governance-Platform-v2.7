"""
core/quality.py — DataQualityChecker

Valida l'input prima dell'inferenza su tre livelli:
  1. Completezza (NaN)
  2. Range (± z_threshold * std dal min/max del training)
  3. Outlier (Z-score > z_threshold rispetto alla media del training)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ethical_governance.infra.metrics import QUALITY_VIOLATIONS
from ethical_governance.infra.observability import get_logger

logger = get_logger(__name__)


class QualityReport(BaseModel):
    passed:        bool
    issues:        List[str]
    quality_score: float
    checked_at:    str


class DataQualityChecker:
    """
    Valida un input rispetto alle statistiche del dataset di training.
    Le statistiche vengono lette dal metadata salvato da ReferenceDataManager.
    """

    def __init__(self, z_threshold: float = 4.0) -> None:
        self.z_threshold = z_threshold

    def check(
        self,
        input_df:   pd.DataFrame,
        ref_meta:   Optional[Dict[str, Any]],
        model_name: str,
    ) -> QualityReport:
        issues: List[str] = []
        col_stats = (ref_meta or {}).get("column_stats", {})

        for col in input_df.columns:
            val = input_df[col].iloc[0]

            if pd.isna(val):
                issues.append(f"'{col}': valore mancante (NaN)")
                QUALITY_VIOLATIONS.labels(model_name=model_name, check="missing").inc()
                continue

            if col not in col_stats:
                continue

            cs   = col_stats[col]
            mean = cs["mean"]
            std  = cs["std"] if cs["std"] > 0 else 1.0

            margin = self.z_threshold * std
            if val < cs["min"] - margin or val > cs["max"] + margin:
                issues.append(
                    f"'{col}': {val:.2f} fuori range "
                    f"[{cs['min'] - margin:.2f}, {cs['max'] + margin:.2f}]"
                )
                QUALITY_VIOLATIONS.labels(model_name=model_name, check="range").inc()

            z = abs((val - mean) / std)
            if z > self.z_threshold:
                issues.append(f"'{col}': Z-score {z:.2f} > soglia {self.z_threshold}")
                QUALITY_VIOLATIONS.labels(model_name=model_name, check="outlier").inc()

        return QualityReport(
            passed=(len(issues) == 0),
            issues=issues,
            quality_score=round(max(0.0, 1.0 - len(issues) * 0.2), 3),
            checked_at=datetime.now(timezone.utc).isoformat(),
        )
