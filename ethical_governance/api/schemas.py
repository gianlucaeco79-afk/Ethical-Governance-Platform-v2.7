"""
api/schemas.py — Modelli Pydantic v2 per request e response.

DataGuard applicato su PredictRequest e QualityCheckRequest:
validator che rifiuta NaN/Inf con errore 422 per campo.
FeedbackRequest valida prediction_id come UUID v4.
"""
from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ethical_governance.config import config


class PredictRequest(BaseModel):
    model_name: str              = Field(..., min_length=1, max_length=200,
                                         examples=["credit_model_v1"])
    features:   Dict[str, float] = Field(
        ..., examples=[{"income": 3500.0, "savings": 12000.0, "gender": 1.0}]
    )
    explain: bool = Field(default=True)

    @field_validator("features")
    @classmethod
    def finite_values(cls, v: Dict[str, float]) -> Dict[str, float]:
        invalid = {
            k: ("NaN" if math.isnan(val) else "+Inf" if val == math.inf else "-Inf")
            for k, val in v.items()
            if not math.isfinite(val)
        }
        if invalid:
            raise ValueError(f"Valori non finiti non ammessi nelle feature: {invalid}.")
        return v

    @field_validator("model_name")
    @classmethod
    def no_path_traversal(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("model_name non puo contenere percorsi.")
        return v


class BatchPredictRequest(BaseModel):
    model_name: str                    = Field(..., min_length=1, max_length=200)
    items:      List[Dict[str, float]] = Field(..., min_length=1)
    explain:    bool                   = Field(default=False)

    @field_validator("items")
    @classmethod
    def check_batch_size(cls, v: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if len(v) > config.BATCH_MAX_SIZE:
            raise ValueError(
                f"Batch troppo grande: {len(v)} > BATCH_MAX_SIZE={config.BATCH_MAX_SIZE}"
            )
        return v

    @field_validator("model_name")
    @classmethod
    def no_path_traversal(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("model_name non puo contenere percorsi.")
        return v


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., min_length=36, max_length=36,
                               description="UUID v4 restituito da /v1/predict")
    model_name:    str = Field(..., min_length=1, max_length=200)
    real_outcome:  int = Field(..., ge=0, le=1, description="0 o 1")

    @field_validator("prediction_id")
    @classmethod
    def valid_uuid(cls, v: str) -> str:
        try:
            parsed = uuid.UUID(v, version=4)
            if str(parsed) != v.lower():
                raise ValueError()
        except (ValueError, AttributeError):
            raise ValueError(f"prediction_id '{v}' non e un UUID v4 valido.")
        return v


class TrainRequest(BaseModel):
    model_name:     str  = Field(..., examples=["logistic_regression"])
    dataset_name:   str  = Field(..., examples=["credit"])
    protected:      str  = Field(..., examples=["gender"])
    target:         str  = Field(..., examples=["approved"])
    use_reweighing: bool = Field(default=True)


class RetriggerRequest(BaseModel):
    model_name: str = Field(..., description="Modello base da ri-addestrare")


class ResolveRequest(BaseModel):
    reviewer_note:  Optional[str] = None
    override_value: Optional[int] = Field(default=None, ge=0, le=1)


class QualityCheckRequest(BaseModel):
    model_name: str              = Field(..., min_length=1)
    features:   Dict[str, float]

    @field_validator("features")
    @classmethod
    def finite_values(cls, v: Dict[str, float]) -> Dict[str, float]:
        invalid = {k: val for k, val in v.items() if not math.isfinite(val)}
        if invalid:
            raise ValueError(f"Valori non finiti: {list(invalid.keys())}")
        return v


class CircuitBreakerRequest(BaseModel):
    action: str = Field(..., pattern="^(trip|reset)$")
