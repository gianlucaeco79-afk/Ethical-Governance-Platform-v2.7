"""
api/routes.py — Tutti gli endpoint FastAPI.

Gli oggetti di servizio vengono letti da app.state,
popolato in main.py durante il lifespan.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ethical_governance.api.dependencies import require_admin, require_tenant
from ethical_governance.infra.exceptions import (
    DomainError,
    FeedbackOwnershipError,
    FeatureMissingError,
    InvalidComparisonRequestError,
    ModelNotFoundError,
    PredictionNotFoundError,
    ReferenceDataMissingError,
    TaskNotFoundError,
)
from ethical_governance.api.schemas import (
    BatchPredictRequest,
    CircuitBreakerRequest,
    FeedbackRequest,
    PredictRequest,
    QualityCheckRequest,
    ResolveRequest,
    RetriggerRequest,
    TrainRequest,
)
from ethical_governance.infra.metrics import FEEDBACK_REJECTED
from ethical_governance.infra.observability import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _s(request: Request) -> Any:
    return request.app.state


def _domain_error_to_http(exc: Exception) -> HTTPException:
    status = 500
    if isinstance(exc, FeatureMissingError):
        status = 400
    elif isinstance(exc, (PredictionNotFoundError, TaskNotFoundError, ReferenceDataMissingError, ModelNotFoundError)):
        status = 404
    elif isinstance(exc, FeedbackOwnershipError):
        status = 403
    elif isinstance(exc, InvalidComparisonRequestError):
        status = 400
    elif isinstance(exc, DomainError):
        status = 400
    return HTTPException(
        status_code=status,
        detail={"error": exc.__class__.__name__, "message": str(exc)},
    )


@router.post("/predict")
async def predict(req: PredictRequest, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    try:
        result = await _s(request).governance.infer(tenant_id, req.model_name, req.features)
        if not req.explain:
            result.pop("explanation", None)
        return result
    except HTTPException:
        raise
    except (DomainError, KeyError) as exc:
        raise _domain_error_to_http(exc)
    except Exception as exc:
        logger.error(f"/v1/predict: {exc}", exc_info=True)
        raise HTTPException(500, "Errore interno.")


@router.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    try:
        result = await _s(request).batch_predictor.run(tenant_id, req.model_name, req.items)
        if not req.explain:
            for item in result.get("results", []):
                item.pop("explanation", None)
        return result
    except HTTPException:
        raise
    except DomainError as exc:
        raise _domain_error_to_http(exc)
    except Exception as exc:
        logger.error(f"/v1/predict/batch: {exc}", exc_info=True)
        raise HTTPException(500, "Errore interno.")


@router.post("/feedback")
async def submit_feedback(req: FeedbackRequest, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    audit = _s(request).audit_log
    event = await audit.find_prediction(req.prediction_id)
    if event is None:
        await audit.log("FEEDBACK_UNKNOWN_PREDICTION", {"tenant_id": tenant_id, "prediction_id": req.prediction_id, "model_name": req.model_name})
        raise _domain_error_to_http(PredictionNotFoundError(f"prediction_id '{req.prediction_id}' non trovato."))
    original_tenant = event.get("tenant_id")
    if original_tenant != tenant_id:
        FEEDBACK_REJECTED.labels(reason="tenant_mismatch").inc()
        await audit.log("FEEDBACK_TENANT_MISMATCH", {"requesting_tenant": tenant_id, "original_tenant": original_tenant, "prediction_id": req.prediction_id, "model_name": req.model_name})
        raise _domain_error_to_http(FeedbackOwnershipError("Accesso negato: prediction_id appartiene a un altro tenant."))
    await _s(request).feedback_store.store(req.prediction_id, req.model_name, req.real_outcome, tenant_id)
    await audit.log("FEEDBACK_RECEIVED", {"tenant_id": tenant_id, "prediction_id": req.prediction_id, "model_name": req.model_name, "real_outcome": req.real_outcome})
    return {"status": "feedback_recorded", "prediction_id": req.prediction_id, "model_name": req.model_name, "real_outcome": req.real_outcome}


@router.get("/feedback/{model_name}")
async def real_accuracy(model_name: str, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    return await _s(request).monitor.compute_real_accuracy(model_name, _s(request).audit_log, _s(request).feedback_store)


@router.post("/retrain/trigger")
async def retrain_trigger(req: RetriggerRequest, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    try:
        from ethical_governance.main import build_demo_dataset
        return await _s(request).retrain_pipeline.trigger(base_model_name=req.model_name, dataset_builder=build_demo_dataset, tenant_id=tenant_id)
    except DomainError as exc:
        raise _domain_error_to_http(exc)
    except Exception as exc:
        logger.error(f"/v1/retrain/trigger: {exc}", exc_info=True)
        raise HTTPException(500, f"Errore retraining: {exc}")


@router.get("/compare")
async def compare_models(request: Request, models: str = Query(...), sample_size: int = Query(default=500, ge=50, le=2000), tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    model_names = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_names) < 2:
        raise _domain_error_to_http(InvalidComparisonRequestError("Specificare almeno 2 modelli separati da virgola."))
    if len(model_names) > 10:
        raise _domain_error_to_http(InvalidComparisonRequestError("Massimo 10 modelli per confronto."))
    try:
        return await _s(request).comparator.compare(model_names, tenant_id, sample_size)
    except (ValueError, DomainError) as exc:
        raise _domain_error_to_http(exc)
    except Exception as exc:
        logger.error(f"/v1/compare: {exc}", exc_info=True)
        raise HTTPException(500, "Errore interno confronto modelli.")


@router.post("/quality-check")
async def quality_check(req: QualityCheckRequest, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    import pandas as pd
    try:
        model    = await _s(request).registry.load(req.model_name)
        ref_meta = _s(request).ref_manager.get_meta(model.dataset_name)
        return _s(request).quality_checker.check(pd.DataFrame([req.features]), ref_meta, req.model_name).model_dump()
    except (DomainError, KeyError) as exc:
        raise _domain_error_to_http(exc)


@router.post("/train")
async def train(req: TrainRequest, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    from ethical_governance.main import build_demo_dataset
    try:
        df      = build_demo_dataset()
        trained = await _s(request).registry.train(df, req.model_name, req.dataset_name, req.protected, req.target, req.use_reweighing)
        return {"status": "trained", "model_name": trained.name, "trained_at": trained.trained_at, "train_metrics": trained.train_metrics, "reference_meta": trained.reference_meta}
    except (ValueError, DomainError) as exc:
        raise _domain_error_to_http(exc)
    except Exception as exc:
        logger.error(f"/v1/train: {exc}", exc_info=True)
        raise HTTPException(500, "Errore interno.")


@router.get("/models")
async def list_models(request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    names   = await _s(request).registry.list_models()
    details: Dict[str, Any] = {}
    for name in names:
        try:
            m = await _s(request).registry.load(name)
            details[name] = {"trained_at": m.trained_at, "dataset_name": m.dataset_name, "features": m.features, "protected": m.protected, "target": m.target, "accuracy": m.train_metrics.get("accuracy")}
        except Exception:
            continue
    return {"models": details}


@router.get("/fairness/{model_name}")
async def fairness_report(model_name: str, request: Request, sample_size: int = 500, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    try:
        model  = await _s(request).registry.load(model_name)
        ref_df = _s(request).ref_manager.get_latest(model.dataset_name)
        if ref_df is None or len(ref_df) < 50:
            raise _domain_error_to_http(ReferenceDataMissingError("Dataset di riferimento insufficiente."))
        sample = ref_df.sample(n=min(sample_size, len(ref_df)), random_state=42)
        limit  = _s(request).tenant_manager.get_fairness_limit(tenant_id)
        return _s(request).fairness_analyzer.analyze(model, sample, limit).model_dump()
    except HTTPException:
        raise
    except (DomainError, KeyError) as exc:
        raise _domain_error_to_http(exc)


@router.get("/hitl/pending")
async def list_pending(request: Request, tenant_id: str = Depends(require_tenant)) -> List[Dict[str, Any]]:
    return [t.model_dump() for t in await _s(request).hitl_manager.list_pending()]


@router.post("/hitl/approve/{task_id}")
async def approve_task(task_id: str, body: ResolveRequest, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    try:
        resolved = await _s(request).hitl_manager.resolve(task_id, True, body.reviewer_note, body.override_value)
        await _s(request).audit_log.log("HUMAN_RESOLUTION", {"tenant_id": tenant_id, "task_id": task_id, "model_key": resolved.model_key, "resolution": "approved", "reviewer_note": body.reviewer_note, "final_decision": resolved.final_decision, "protected_value": resolved.input_data.get("gender")})
        return {"status": "approved", "task": resolved.model_dump()}
    except (DomainError, KeyError) as exc:
        raise _domain_error_to_http(exc)


@router.post("/hitl/reject/{task_id}")
async def reject_task(task_id: str, body: ResolveRequest, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    try:
        resolved = await _s(request).hitl_manager.resolve(task_id, False, body.reviewer_note, body.override_value)
        await _s(request).audit_log.log("HUMAN_RESOLUTION", {"tenant_id": tenant_id, "task_id": task_id, "model_key": resolved.model_key, "resolution": "overridden", "reviewer_note": body.reviewer_note, "final_decision": resolved.final_decision, "protected_value": resolved.input_data.get("gender")})
        return {"status": "overridden", "task": resolved.model_dump()}
    except (DomainError, KeyError) as exc:
        raise _domain_error_to_http(exc)


@router.get("/corrections/{model_name}")
async def auto_correction_report(model_name: str, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    return await _s(request).auto_correction.generate_report(model_name)


@router.get("/monitor/{model_name}")
async def monitor_history(model_name: str, request: Request, last_n: int = 100, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    history = await _s(request).monitor.get_history(model_name, last_n=last_n)
    try:
        model    = await _s(request).registry.load(model_name)
        baseline = model.train_metrics.get("accuracy", 1.0)
    except (DomainError, KeyError):
        baseline = 1.0
    return {"history": history, "degradation": await _s(request).monitor.degradation_report(model_name, baseline)}


@router.get("/audit")
async def audit_trail(request: Request, n: int = 50, event_type: Optional[str] = None, tenant_id: str = Depends(require_tenant)) -> List[Dict[str, Any]]:
    return await _s(request).audit_log.read_last(n, event_type=event_type)


@router.get("/reference/{dataset_name}")
async def reference_meta(dataset_name: str, request: Request, tenant_id: str = Depends(require_tenant)) -> Dict[str, Any]:
    meta = _s(request).ref_manager.get_meta(dataset_name)
    if not meta:
        raise _domain_error_to_http(ReferenceDataMissingError(f"Nessun dataset per '{dataset_name}'."))
    return meta


@router.post("/admin/cache/clear")
async def clear_cache(request: Request, model_name: Optional[str] = None, _: None = Depends(require_admin)) -> Dict[str, Any]:
    return await _s(request).registry.clear_cache(model_name)


@router.post("/admin/circuit-breaker/{target_tenant}")
async def manage_circuit_breaker(target_tenant: str, body: CircuitBreakerRequest, request: Request, _: None = Depends(require_admin)) -> Dict[str, str]:
    if body.action == "trip":
        await _s(request).circuit_breaker.trip(target_tenant)
        return {"status": f"Circuit breaker attivato per '{target_tenant}'."}
    await _s(request).circuit_breaker.reset(target_tenant)
    return {"status": f"Circuit breaker resettato per '{target_tenant}'."}


@router.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    from ethical_governance.ml.explainability import SHAP_AVAILABLE
    state = _s(request)
    return {
        "status": "ok", "version": "2.7.0", "env": state.config.ENV,
        "models": await state.registry.list_models(),
        "shap_available": SHAP_AVAILABLE,
        "redis": "connected" if state.redis_client else "disabled",
        "risk_weights": {"fairness": state.config.RISK_WEIGHT_FAIRNESS, "drift": state.config.RISK_WEIGHT_DRIFT, "quality": state.config.RISK_WEIGHT_QUALITY},
        "alert_webhook": bool(state.config.ALERT_WEBHOOK_URL),
        "batch_max_size": state.config.BATCH_MAX_SIZE,
    }


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
