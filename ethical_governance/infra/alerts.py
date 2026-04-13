"""
infra/alerts.py — AlertService e WebhookNotifier

Notifiche push proattive al superamento di soglie configurabili.
Webhook via httpx se ALERT_WEBHOOK_URL e' impostato, log strutturato come fallback.
"""
from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import httpx
from pydantic import BaseModel, Field

from ethical_governance.config import config
from ethical_governance.infra.metrics import ALERTS_FIRED
from ethical_governance.infra.observability import get_logger

if TYPE_CHECKING:
    from ethical_governance.infra.audit import AuditLoggerBase
    from ethical_governance.core.fairness import FairnessReport

logger = get_logger(__name__)


class AlertPayload(BaseModel):
    """Payload strutturato inviato al webhook o loggato come fallback."""
    alert_id:   str
    rule:       str
    severity:   str
    model_name: str
    tenant_id:  str
    value:      float
    threshold:  float
    message:    str
    timestamp:  str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class NotifierBase(ABC):
    """Interfaccia astratta per un notifier di alert."""

    @abstractmethod
    async def send(self, payload: AlertPayload) -> bool:
        raise NotImplementedError


class WebhookNotifier(NotifierBase):
    """
    Invia AlertPayload via HTTP POST al webhook configurato.
    Fallback silenzioso a log strutturato se il webhook fallisce.
    """

    def __init__(self, webhook_url: Optional[str], timeout: float = 5.0) -> None:
        self._url     = webhook_url
        self._timeout = timeout

    async def send(self, payload: AlertPayload) -> bool:
        if not self._url:
            return False
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, json=payload.model_dump())
                if resp.status_code < 300:
                    return True
                logger.warning(f"Alert webhook HTTP {resp.status_code}: {self._url}")
                return False
        except Exception as exc:
            logger.warning(f"Alert webhook fallito ({type(exc).__name__}): {exc}")
            return False


class AlertService:
    """
    Valuta tre regole dopo ogni inferenza e notifica proattivamente.
    Non solleva mai eccezioni - e' un effetto collaterale non bloccante.
    """

    def __init__(self, notifier: WebhookNotifier, audit: "AuditLoggerBase") -> None:
        self._notifier = notifier
        self._audit    = audit

    async def evaluate_and_fire(
        self,
        model_name:      str,
        tenant_id:       str,
        fairness_report: Optional["FairnessReport"],
        drift:           Dict[str, Dict[str, Any]],
        risk_score:      float,
        request_id:      Optional[str] = None,
    ) -> List[str]:
        fired: List[str] = []
        rules: List[tuple] = []

        if fairness_report and config.ALERT_DPD_THRESHOLD > 0:
            dpd = abs(fairness_report.demographic_parity_difference)
            if dpd >= config.ALERT_DPD_THRESHOLD:
                rules.append(("HIGH_DPD", fairness_report.severity, dpd, config.ALERT_DPD_THRESHOLD))

        if config.ALERT_DRIFT_THRESHOLD > 0:
            n_drifted = sum(1 for v in drift.values() if v.get("drift_detected"))
            if n_drifted >= config.ALERT_DRIFT_THRESHOLD:
                rules.append(("DRIFT_THRESHOLD", "HIGH", float(n_drifted), float(config.ALERT_DRIFT_THRESHOLD)))

        if config.ALERT_RISK_THRESHOLD > 0 and risk_score >= config.ALERT_RISK_THRESHOLD:
            rules.append(("HIGH_RISK", "CRITICAL", risk_score, config.ALERT_RISK_THRESHOLD))

        for rule, severity, value, threshold in rules:
            alert_id = str(uuid.uuid4())
            payload  = AlertPayload(
                alert_id=alert_id, rule=rule, severity=severity,
                model_name=model_name, tenant_id=tenant_id,
                value=round(value, 4), threshold=threshold,
                message=f"{rule}: valore {value:.4f} >= soglia {threshold:.4f} (modello={model_name})",
            )
            sent = await self._notifier.send(payload)
            if not sent:
                logger.warning(f"ALERT [{rule}] severity={severity} model={model_name} value={value:.4f}")
            asyncio.create_task(
                self._audit.log("ALERT_FIRED", {
                    "alert_id":    alert_id,
                    "rule":        rule,
                    "severity":    severity,
                    "model_name":  model_name,
                    "tenant_id":   tenant_id,
                    "value":       round(value, 4),
                    "threshold":   threshold,
                    "webhook_sent": sent,
                    "request_id":  request_id,
                }),
                name=f"alert_{alert_id}",
            )
            ALERTS_FIRED.labels(rule=rule, severity=severity).inc()
            fired.append(alert_id)

        return fired
