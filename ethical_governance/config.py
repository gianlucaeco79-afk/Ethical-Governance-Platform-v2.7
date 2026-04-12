"""
config.py — Configurazione centralizzata letta da .env / variabili d'ambiente.
Nessun segreto hardcoded. model_validator blocca avvio in produzione con
chiavi di default o pesi di rischio che non sommano a 1.0.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from pydantic import BaseModel, Field, field_validator, model_validator


class Settings(BaseModel):
    ENV:      str  = Field(default_factory=lambda: os.getenv("ENV", "development"))
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(
            os.getenv("GOV_BASE_DIR", "./governance_system")
        ).expanduser().resolve()
    )
    REDIS_URL: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_URL"))

    FAIRNESS_LIMIT:   float = Field(default_factory=lambda: float(os.getenv("FAIRNESS_LIMIT",   "0.10")))
    DRIFT_P_VALUE:    float = Field(default_factory=lambda: float(os.getenv("DRIFT_P_VALUE",    "0.05")))
    DRIFT_MIN_WINDOW: int   = Field(default_factory=lambda: int(  os.getenv("DRIFT_MIN_WINDOW", "20")))

    QUALITY_Z_THRESHOLD: float = Field(
        default_factory=lambda: float(os.getenv("QUALITY_Z_THRESHOLD", "4.0"))
    )

    AUTO_CORRECT_MIN_SAMPLES:  int = Field(
        default_factory=lambda: int(os.getenv("AUTO_CORRECT_MIN_SAMPLES", "10"))
    )
    MIN_CORRECTIONS_THRESHOLD: int = Field(
        default_factory=lambda: int(os.getenv("MIN_CORRECTIONS_THRESHOLD", "10"))
    )

    HITL_REQUIRED_LEVELS: List[str] = ["HIGH", "CRITICAL", "UNACCEPTABLE"]
    SUPPORTED_MODELS:     List[str] = [
        "logistic_regression", "random_forest", "gradient_boosting"
    ]

    TENANT_PLANS: Dict[str, Dict[str, Any]] = {
        "premium":  {"rate_limit": 500, "max_models": 50, "fairness_limit": 0.05},
        "standard": {"rate_limit": 100, "max_models": 20, "fairness_limit": 0.10},
        "free":     {"rate_limit": 20,  "max_models": 5,  "fairness_limit": 0.15},
    }
    DEFAULT_RATE_LIMIT: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_RATE_LIMIT", "100"))
    )
    ADMIN_API_KEY: str = Field(
        default_factory=lambda: os.getenv("ADMIN_API_KEY", "admin-key-change-me")
    )

    RISK_WEIGHT_FAIRNESS:       float = Field(default_factory=lambda: float(os.getenv("RISK_WEIGHT_FAIRNESS", "0.50")))
    RISK_WEIGHT_DRIFT:          float = Field(default_factory=lambda: float(os.getenv("RISK_WEIGHT_DRIFT",    "0.30")))
    RISK_WEIGHT_QUALITY:        float = Field(default_factory=lambda: float(os.getenv("RISK_WEIGHT_QUALITY",  "0.20")))
    RISK_WEIGHT_SEVERITY_BOOST: float = Field(default_factory=lambda: float(os.getenv("RISK_WEIGHT_SEVERITY_BOOST", "0.30")))

    BATCH_MAX_SIZE: int = Field(
        default_factory=lambda: int(os.getenv("BATCH_MAX_SIZE", "50"))
    )

    ALERT_WEBHOOK_URL:       Optional[str] = Field(default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL"))
    ALERT_WEBHOOK_TIMEOUT_S: float         = Field(default_factory=lambda: float(os.getenv("ALERT_WEBHOOK_TIMEOUT_S", "5.0")))
    ALERT_DPD_THRESHOLD:     float         = Field(default_factory=lambda: float(os.getenv("ALERT_DPD_THRESHOLD",   "0.15")))
    ALERT_DRIFT_THRESHOLD:   int           = Field(default_factory=lambda: int(  os.getenv("ALERT_DRIFT_THRESHOLD", "3")))
    ALERT_RISK_THRESHOLD:    float         = Field(default_factory=lambda: float(os.getenv("ALERT_RISK_THRESHOLD",  "0.60")))

    @field_validator("FAIRNESS_LIMIT", "DRIFT_P_VALUE")
    @classmethod
    def must_be_ratio(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("Deve essere in (0, 1)")
        return v

    @field_validator("QUALITY_Z_THRESHOLD")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Deve essere > 0")
        return v

    @field_validator("RISK_WEIGHT_FAIRNESS", "RISK_WEIGHT_DRIFT", "RISK_WEIGHT_QUALITY")
    @classmethod
    def must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("I pesi di rischio devono essere >= 0")
        return v

    @model_validator(mode="after")
    def validate_all(self) -> "Settings":
        total = self.RISK_WEIGHT_FAIRNESS + self.RISK_WEIGHT_DRIFT + self.RISK_WEIGHT_QUALITY
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"RISK_WEIGHT_FAIRNESS + RISK_WEIGHT_DRIFT + RISK_WEIGHT_QUALITY "
                f"deve sommare a 1.0 (attuale: {total:.6f})"
            )
        if self.ENV.lower() == "production":
            if self.ADMIN_API_KEY == "admin-key-change-me":
                raise ValueError("ADMIN_API_KEY deve essere impostata in produzione.")
            if not self.REDIS_URL:
                raise ValueError("REDIS_URL è obbligatoria in produzione.")
        return self

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    model_config = {"arbitrary_types_allowed": True}


try:
    config = Settings()
except Exception as exc:
    raise SystemExit(f"FATAL: configurazione non valida — {exc}") from exc

for _d in ["models", "data_reference", "pending_tasks",
           "serving_buffer", "monitor", "feedback", "retrain"]:
    (config.BASE_DIR / _d).mkdir(parents=True, exist_ok=True)
