"""
api/dependencies.py — Dipendenze FastAPI (require_tenant, require_admin).
"""
from __future__ import annotations

from fastapi import Header, HTTPException

from ethical_governance.config import config
from ethical_governance.infra.tenancy import TenantManager

_tenant_manager: TenantManager | None = None


def set_tenant_manager(tm: TenantManager) -> None:
    global _tenant_manager
    _tenant_manager = tm


async def require_tenant(
    tenant_id: str = Header(..., alias="X-Tenant-Id"),
    api_key:   str = Header(..., alias="X-API-Key"),
) -> str:
    if _tenant_manager is None:
        raise RuntimeError("TenantManager non inizializzato.")
    if not await _tenant_manager.validate(tenant_id, api_key):
        raise HTTPException(403, "Tenant ID o API Key non validi.")
    return tenant_id


async def require_admin(
    admin_key: str = Header(..., alias="X-Admin-Key"),
) -> None:
    if admin_key != config.ADMIN_API_KEY:
        raise HTTPException(403, "Chiave admin non valida.")
