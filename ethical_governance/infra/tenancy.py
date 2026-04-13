"""
infra/tenancy.py — TenantManager, RateLimiter, CircuitBreaker

TenantManager:  validazione API key da env, piani e soglie per tenant.
RateLimiter:    finestra scorrevole (Redis distribuito o fallback in-memory).
CircuitBreaker: isola tenant anomali con TTL (Redis o fallback in-memory).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from ethical_governance.config import config
from ethical_governance.infra.metrics import CB_TRIPS
from ethical_governance.infra.observability import get_logger

logger = get_logger(__name__)


class TenantManager:
    """
    Gestisce tenant, piani e API key.
    Le chiavi sono lette da variabili d'ambiente — nessun valore hardcoded.
    In produzione i fallback di sviluppo non vengono mai usati.
    """

    _KEY_ENV_VARS: Dict[str, str] = {
        "premium_demo":  "TENANT_PREMIUM_API_KEY",
        "standard_demo": "TENANT_STANDARD_API_KEY",
        "free_demo":     "TENANT_FREE_API_KEY",
    }
    _DEV_FALLBACKS: Dict[str, str] = {
        "premium_demo":  "premium-key-123",
        "standard_demo": "standard-key-456",
        "free_demo":     "free-key-789",
    }

    def __init__(self) -> None:
        self._plans: Dict[str, str] = {
            "premium_demo":  "premium",
            "standard_demo": "standard",
            "free_demo":     "free",
        }

    def _get_api_key(self, tenant_id: str) -> Optional[str]:
        env_var = self._KEY_ENV_VARS.get(tenant_id)
        if env_var:
            key = os.getenv(env_var)
            if key:
                return key
        if not config.is_production:
            return self._DEV_FALLBACKS.get(tenant_id)
        return None

    async def validate(self, tenant_id: str, api_key: str) -> bool:
        expected = self._get_api_key(tenant_id)
        return bool(expected and expected == api_key)

    def get_plan(self, tenant_id: str) -> str:
        return self._plans.get(tenant_id, "free")

    def get_rate_limit(self, tenant_id: str) -> int:
        return config.TENANT_PLANS.get(self.get_plan(tenant_id), {}).get("rate_limit", config.DEFAULT_RATE_LIMIT)

    def get_fairness_limit(self, tenant_id: str) -> float:
        return config.TENANT_PLANS.get(self.get_plan(tenant_id), {}).get("fairness_limit", config.FAIRNESS_LIMIT)


class RateLimiter:
    """Rate limiting a finestra scorrevole (60s). Redis o in-memory."""

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        self._redis   = redis_client
        self._windows: Dict[str, List[float]] = {}
        self._lock    = asyncio.Lock()

    async def is_allowed(self, tenant_id: str, limit: int) -> bool:
        return (
            await self._redis_check(tenant_id, limit)
            if self._redis
            else await self._memory_check(tenant_id, limit)
        )

    async def _redis_check(self, tenant_id: str, limit: int) -> bool:
        bucket = int(time.time()) // 60
        key    = f"rate:{tenant_id}:{bucket}"
        count  = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, 61)
        return count <= limit

    async def _memory_check(self, tenant_id: str, limit: int) -> bool:
        now    = time.time()
        cutoff = now - 60.0
        async with self._lock:
            w = self._windows.setdefault(tenant_id, [])
            while w and w[0] < cutoff:
                w.pop(0)
            if len(w) >= limit:
                return False
            w.append(now)
            return True


class CircuitBreaker:
    """
    Isola tenant che generano troppe anomalie (TTL = 300s).
    Redis per distribuzione multi-processo, dict in-memory come fallback.
    """

    _TTL = 300

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        self._redis = redis_client
        self._state: Dict[str, float] = {}

    async def is_open(self, tenant_id: str) -> bool:
        if self._redis:
            return bool(await self._redis.get(f"cb:{tenant_id}"))
        expiry = self._state.get(tenant_id, 0.0)
        if time.time() > expiry:
            self._state.pop(tenant_id, None)
            return False
        return True

    async def trip(self, tenant_id: str) -> None:
        CB_TRIPS.labels(tenant_id=tenant_id).inc()
        if self._redis:
            await self._redis.setex(f"cb:{tenant_id}", self._TTL, "1")
        else:
            self._state[tenant_id] = time.time() + self._TTL

    async def reset(self, tenant_id: str) -> None:
        if self._redis:
            await self._redis.delete(f"cb:{tenant_id}")
        else:
            self._state.pop(tenant_id, None)
