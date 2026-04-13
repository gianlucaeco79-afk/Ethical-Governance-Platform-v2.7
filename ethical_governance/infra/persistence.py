"""
infra/persistence.py — ReferenceDataManager e ServingBuffer

ReferenceDataManager: dataset di riferimento versionati con hash SHA-256.
ServingBuffer: buffer rolling per drift detection (Redis o JSONL).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ethical_governance.infra.observability import get_logger

logger = get_logger(__name__)


class ReferenceDataManager:
    """
    Gestisce dataset di riferimento versionati con indice JSON e hash SHA-256.
    Le statistiche per colonna vengono salvate nel metadata per DataQualityChecker.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir  = data_dir
        self._idx_path = data_dir / "reference_index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if not self._idx_path.exists():
            return {}
        try:
            return json.loads(self._idx_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_index(self) -> None:
        self._idx_path.write_text(
            json.dumps(self._index, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _hash(df: pd.DataFrame) -> str:
        return hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()

    def save(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        version  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{version}.parquet"
        df.to_parquet(self.data_dir / filename)
        meta: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "version":      version,
            "hash":         self._hash(df),
            "created_at":   datetime.now(timezone.utc).isoformat(),
            "rows":         len(df),
            "cols":         len(df.columns),
            "file":         filename,
            "column_stats": {
                col: {
                    "mean": float(df[col].mean()),
                    "std":  float(df[col].std()),
                    "min":  float(df[col].min()),
                    "max":  float(df[col].max()),
                }
                for col in df.select_dtypes(include=[np.number]).columns
            },
        }
        self._index[f"{dataset_name}:{version}"] = meta
        self._save_index()
        return meta

    def get_latest(self, dataset_name: str) -> Optional[pd.DataFrame]:
        meta = self.get_meta(dataset_name)
        if not meta:
            return None
        path = self.data_dir / meta["file"]
        return pd.read_parquet(path) if path.exists() else None

    def get_meta(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        candidates = [
            v for k, v in self._index.items()
            if k.startswith(f"{dataset_name}:")
        ]
        return sorted(candidates, key=lambda x: x["version"])[-1] if candidates else None


class ServingBuffer:
    """
    Buffer rolling delle ultime N richieste di serving.
    Usato dal DriftDetector per confrontare distribuzione reale vs training.
    Redis se disponibile, JSONL come fallback.
    """

    def __init__(self, buffer_dir: Path, redis_client: Optional[Any], max_size: int) -> None:
        self.buffer_dir = buffer_dir
        self._redis     = redis_client
        self.max_size   = max_size
        self._lock      = asyncio.Lock()

    async def append(self, dataset_name: str, row: Dict[str, Any]) -> None:
        if self._redis:
            key = f"serving:{dataset_name}"
            await self._redis.lpush(key, json.dumps(row, default=str))
            await self._redis.ltrim(key, 0, self.max_size - 1)
        else:
            path = self.buffer_dir / f"{dataset_name}.jsonl"
            async with self._lock:
                await asyncio.to_thread(self._append_sync, path, row)

    def _append_sync(self, path: Path, row: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > self.max_size:
            path.write_text("\n".join(lines[-self.max_size:]) + "\n", encoding="utf-8")

    async def load(self, dataset_name: str) -> pd.DataFrame:
        if self._redis:
            rows_raw = await self._redis.lrange(f"serving:{dataset_name}", 0, -1)
            return pd.DataFrame([json.loads(r) for r in rows_raw]) if rows_raw else pd.DataFrame()
        path = self.buffer_dir / f"{dataset_name}.jsonl"
        if not path.exists():
            return pd.DataFrame()
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
