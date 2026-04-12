"""
main.py — Entry point di Ethical Governance Platform v2.7

Wiring dei componenti, lifespan FastAPI, middleware rate-limit.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None  # type: ignore

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ethical_governance.api.dependencies import set_tenant_manager
from ethical_governance.infra.exceptions import ModelNotFoundError
from ethical_governance.api.routes import router
from ethical_governance.config import config
from ethical_governance.core.drift import DriftDetector
from ethical_governance.core.fairness import FairnessAnalyzer
from ethical_governance.core.quality import DataQualityChecker
from ethical_governance.core.risk import RiskEngine
from ethical_governance.governance.batch import BatchPredictor
from ethical_governance.governance.comparison import ModelComparator
from ethical_governance.governance.correction import AutoCorrectionService
from ethical_governance.governance.engine import GovernanceEngine
from ethical_governance.governance.hitl import HITLManager
from ethical_governance.infra.alerts import AlertService, WebhookNotifier
from ethical_governance.infra.audit import AuditLogger, FeedbackStore
from ethical_governance.infra.observability import get_logger
from ethical_governance.infra.persistence import ReferenceDataManager, ServingBuffer
from ethical_governance.infra.tenancy import CircuitBreaker, RateLimiter, TenantManager
from ethical_governance.ml.explainability import ExplainabilityService
from ethical_governance.ml.models import ModelRegistry, TrainedModel
from ethical_governance.ml.monitor import ModelMonitor
from ethical_governance.ml.retraining import RetrainingPipeline

logger = get_logger(__name__)


def build_demo_dataset(n: int = 600):
    """Dataset sintetico di concessione credito con bias su 'gender'."""
    import numpy as np
    import pandas as pd
    rng     = np.random.default_rng(42)
    income  = rng.normal(3000.0, 1000.0, n)
    savings = rng.normal(10000.0, 5000.0, n)
    gender  = rng.integers(0, 2, n)
    noise   = rng.normal(0.0, 1.0, n)
    score   = 0.002 * income + 0.0002 * savings + 0.6 * gender + 0.2 * noise
    return pd.DataFrame({
        "income":   income,
        "savings":  savings,
        "gender":   gender,
        "approved": (score > score.mean()).astype(int),
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = app.state

    state.redis_client: Optional[Any] = None
    if REDIS_AVAILABLE and config.REDIS_URL:
        try:
            state.redis_client = aioredis.from_url(
                config.REDIS_URL, decode_responses=True
            )
            logger.info("Redis connesso.")
        except Exception as exc:
            logger.warning(f"Redis non disponibile ({exc}).")

    rc = state.redis_client
    state.config = config

    state.audit_log      = AuditLogger(config.BASE_DIR / "audit_trail.jsonl")
    state.feedback_store = FeedbackStore(config.BASE_DIR / "feedback" / "feedback.jsonl")
    state.ref_manager    = ReferenceDataManager(config.BASE_DIR / "data_reference")
    state.serving_buffer = ServingBuffer(config.BASE_DIR / "serving_buffer", rc, config.DRIFT_MIN_WINDOW * 3)
    state.tenant_manager  = TenantManager()
    state.rate_limiter    = RateLimiter(rc)
    state.circuit_breaker = CircuitBreaker(rc)

    set_tenant_manager(state.tenant_manager)

    webhook_notifier = WebhookNotifier(config.ALERT_WEBHOOK_URL, config.ALERT_WEBHOOK_TIMEOUT_S)
    state.alert_service = AlertService(webhook_notifier, state.audit_log)

    state.registry = ModelRegistry(config.BASE_DIR / "models", state.ref_manager, rc)
    state.monitor  = ModelMonitor(config.BASE_DIR / "monitor")

    state.fairness_analyzer = FairnessAnalyzer()
    state.drift_detector    = DriftDetector(config.DRIFT_P_VALUE)
    state.quality_checker   = DataQualityChecker(config.QUALITY_Z_THRESHOLD)
    state.risk_engine       = RiskEngine(state.circuit_breaker)
    state.explainability    = ExplainabilityService()

    state.hitl_manager    = HITLManager(config.BASE_DIR / "pending_tasks", rc)
    state.auto_correction = AutoCorrectionService(state.audit_log)

    state.governance = GovernanceEngine(
        registry=state.registry, ref_mgr=state.ref_manager,
        audit=state.audit_log, buffer=state.serving_buffer,
        fairness=state.fairness_analyzer, drift=state.drift_detector,
        quality=state.quality_checker, risk=state.risk_engine,
        hitl=state.hitl_manager, cb=state.circuit_breaker,
        monitor=state.monitor, explainability=state.explainability,
        feedback_store=state.feedback_store, alert_service=state.alert_service,
        tenant_manager=state.tenant_manager,
    )

    state.batch_predictor  = BatchPredictor(state.governance)
    state.comparator       = ModelComparator(state.registry, state.ref_manager, state.fairness_analyzer, state.tenant_manager)
    state.retrain_pipeline = RetrainingPipeline(registry=state.registry, auto_correction=state.auto_correction, audit=state.audit_log, retrain_dir=config.BASE_DIR / "retrain")

    demo_key = "credit_model_v1"
    try:
        await state.registry.load(demo_key)
        logger.info(f"Demo '{demo_key}' caricato dalla cache.")
    except (ModelNotFoundError, KeyError):
        logger.info(f"Pre-addestramento demo '{demo_key}'...")
        df      = build_demo_dataset()
        trained = await state.registry.train(df, "logistic_regression", "credit", "gender", "approved", True)
        stable  = TrainedModel(
            name=demo_key, pipeline=trained.pipeline, features=trained.features,
            protected=trained.protected, target=trained.target, dataset_name=trained.dataset_name,
            train_metrics=trained.train_metrics, reference_meta=trained.reference_meta,
        )
        await state.registry.save(stable)
        logger.info(f"Demo pronto. Accuracy={trained.train_metrics.get('accuracy', 0):.4f}")

    yield

    logger.info("Shutdown Ethical Governance Platform v2.7.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Ethical Governance Platform",
        version="2.7.0",
        description="AI Fairness Infrastructure v2.7",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next: Any) -> Any:
        tenant_id = request.headers.get("X-Tenant-Id")
        if tenant_id and hasattr(request.app.state, "rate_limiter"):
            tm    = request.app.state.tenant_manager
            rl    = request.app.state.rate_limiter
            limit = tm.get_rate_limit(tenant_id)
            if not await rl.is_allowed(tenant_id, limit):
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Rate limit superato per '{tenant_id}' ({limit} req/min)."},
                )
        return await call_next(request)

    app.include_router(router, prefix="/v1")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ethical_governance.main:app", host="0.0.0.0", port=8000, reload=config.ENV != "production", log_level="info")
