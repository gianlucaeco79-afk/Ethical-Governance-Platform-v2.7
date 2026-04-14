"""
ml/models.py — TrainedModel e ModelRegistry

ModelRegistry con Lazy Loading + Cache in-memory + clear_cache().
Training con reweighing probabilistico corretto e weight_multiplier opzionale.
Le eccezioni di dominio vengono propagate come ModelNotFoundError quando un
modello richiesto non esiste.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ethical_governance.config import config
from ethical_governance.infra.exceptions import ModelNotFoundError
from ethical_governance.infra.metrics import CACHE_HITS, CACHE_MISSES
from ethical_governance.infra.observability import get_logger
from ethical_governance.infra.persistence import ReferenceDataManager

logger = get_logger(__name__)


class TrainedModel:
    """Wrapper tipizzato attorno a un Pipeline sklearn."""

    def __init__(
        self,
        name:           str,
        pipeline:       Pipeline,
        features:       List[str],
        protected:      str,
        target:         str,
        dataset_name:   str,
        trained_at:     Optional[str]            = None,
        train_metrics:  Optional[Dict[str, Any]] = None,
        reference_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name           = name
        self.pipeline       = pipeline
        self.features       = features
        self.protected      = protected
        self.target         = target
        self.dataset_name   = dataset_name
        self.trained_at     = trained_at or datetime.now(timezone.utc).isoformat()
        self.train_metrics  = train_metrics  or {}
        self.reference_meta = reference_meta or {}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(df[self.features])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(df[self.features])[:, 1]
        s = self.pipeline.decision_function(df[self.features])
        return 1.0 / (1.0 + np.exp(-s))


_ALGO_MAP: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}


class ModelRegistry:
    """
    Registro modelli con Lazy Loading + Cache.
    Prima chiamata a load() -> carica da disco (cache miss).
    Chiamate successive -> restituisce l'istanza dalla cache (cache hit).
    """

    def __init__(self, model_dir: Path, ref_manager: ReferenceDataManager,
                 redis_client: Optional[Any] = None) -> None:
        self.model_dir = model_dir
        self._ref_mgr  = ref_manager
        self._redis    = redis_client
        self._cache:   Dict[str, TrainedModel] = {}
        self._lock     = asyncio.Lock()

    def _path(self, name: str) -> Path:
        return self.model_dir / f"{name}.joblib"

    @staticmethod
    def _build_pipeline(model_name: str) -> Pipeline:
        if model_name not in _ALGO_MAP:
            raise ValueError(f"'{model_name}' non supportato: {list(_ALGO_MAP.keys())}")
        return Pipeline([("scaler", StandardScaler()), ("clf", clone(_ALGO_MAP[model_name]))])

    @staticmethod
    def _compute_weights(df: pd.DataFrame, target: str, protected: str,
                         weight_multiplier: float = 1.0) -> np.ndarray:
        weights = np.ones(len(df), dtype=float)
        eps = 1e-6
        for y_val in df[target].unique():
            for a_val in df[protected].unique():
                mask = (df[target] == y_val) & (df[protected] == a_val)
                if not mask.any():
                    continue
                p_y  = float((df[target]    == y_val).mean())
                p_a  = float((df[protected] == a_val).mean())
                p_ya = float(mask.mean())
                w    = (p_y * p_a) / (p_ya + eps) * weight_multiplier
                weights[mask.values] = float(np.clip(w, 0.2, 5.0))
        return weights

    async def load(self, name: str) -> TrainedModel:
        async with self._lock:
            if name in self._cache:
                CACHE_HITS.labels(model_name=name).inc()
                return self._cache[name]
            CACHE_MISSES.labels(model_name=name).inc()
            path = self._path(name)
            if not path.exists():
                raise ModelNotFoundError(f"Modello '{name}' non trovato.")
            model: TrainedModel = await asyncio.to_thread(joblib.load, path)
            self._cache[name] = model
            logger.info(f"Modello '{name}' caricato da disco.")
            return model

    async def save(self, model: TrainedModel) -> None:
        async with self._lock:
            await asyncio.to_thread(joblib.dump, model, self._path(model.name))
            self._cache[model.name] = model

    async def clear_cache(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        async with self._lock:
            if model_name is None:
                cleared = list(self._cache.keys())
                self._cache.clear()
                logger.info(f"Cache svuotata: {len(cleared)} modelli rimossi.")
                return {"cleared": cleared, "total": len(cleared)}
            if model_name in self._cache:
                del self._cache[model_name]
                logger.info(f"Cache svuotata per '{model_name}'.")
                return {"cleared": [model_name], "total": 1}
            return {"cleared": [], "total": 0, "note": f"'{model_name}' non in cache"}

    async def list_models(self) -> List[str]:
        paths = list(self.model_dir.glob("*.joblib"))
        return sorted({p.stem for p in paths} | set(self._cache.keys()))

    async def train(
        self,
        df:                pd.DataFrame,
        model_name:        str,
        dataset_name:      str,
        protected:         str,
        target:            str,
        use_reweighing:    bool  = True,
        weight_multiplier: float = 1.0,
    ) -> TrainedModel:
        if model_name not in config.SUPPORTED_MODELS:
            raise ValueError(f"'{model_name}' non supportato: {config.SUPPORTED_MODELS}")

        features = [c for c in df.columns if c != target]
        X, y     = df[features].copy(), df[target].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline    = self._build_pipeline(model_name)
        fit_kwargs: Dict[str, Any] = {}
        if use_reweighing:
            train_df = pd.concat(
                [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
            )
            fit_kwargs["clf__sample_weight"] = self._compute_weights(
                train_df, target, protected, weight_multiplier
            )

        await asyncio.to_thread(pipeline.fit, X_train, y_train, **fit_kwargs)

        y_pred   = pipeline.predict(X_test)
        metrics: Dict[str, Any] = {
            "accuracy":         float(accuracy_score(y_test, y_pred)),
            "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
            "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report":           classification_report(y_test, y_pred, output_dict=True),
            "train_size":       len(X_train),
            "test_size":        len(X_test),
        }

        ref_meta = self._ref_mgr.save(df, dataset_name)
        key      = f"{dataset_name}_{model_name}_{ref_meta['version']}"
        trained  = TrainedModel(
            name=key, pipeline=pipeline, features=features,
            protected=protected, target=target, dataset_name=dataset_name,
            train_metrics=metrics, reference_meta=ref_meta,
        )
        await self.save(trained)
        logger.info(f"Modello '{key}' pronto. Accuracy={metrics['accuracy']:.4f} Reweighing={use_reweighing}")
        return trained
