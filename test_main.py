"""
test_main.py — Suite di test per Ethical Governance Platform v2.7
==================================================================
Architettura modulare: ogni test importa solo il componente che testa.

Esecuzione:
    pytest test_main.py -v
    pytest test_main.py -m unit
    pytest test_main.py -m integration
    pytest test_main.py -m api
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_TMPDIR = tempfile.mkdtemp()
os.environ.setdefault("ENV",                   "development")
os.environ.setdefault("GOV_BASE_DIR",          _TMPDIR)
os.environ.setdefault("ADMIN_API_KEY",         "test-admin-key")
os.environ.setdefault("RISK_WEIGHT_FAIRNESS",  "0.50")
os.environ.setdefault("RISK_WEIGHT_DRIFT",     "0.30")
os.environ.setdefault("RISK_WEIGHT_QUALITY",   "0.20")
os.environ.setdefault("BATCH_MAX_SIZE",        "10")
os.environ.setdefault("ALERT_DPD_THRESHOLD",   "0.15")
os.environ.setdefault("ALERT_DRIFT_THRESHOLD", "3")
os.environ.setdefault("ALERT_RISK_THRESHOLD",  "0.60")

from ethical_governance.config import config
from ethical_governance.core.drift import DriftDetector
from ethical_governance.core.fairness import FairnessAnalyzer
from ethical_governance.core.quality import DataQualityChecker
from ethical_governance.core.risk import RiskEngine
from ethical_governance.governance.correction import AutoCorrectionService
from ethical_governance.governance.hitl import DecisionTask, HITLManager
from ethical_governance.infra.alerts import AlertService, WebhookNotifier
from ethical_governance.infra.audit import AuditLogger, FeedbackStore
from ethical_governance.infra.observability import MockTracer, OTelTracer, observability
from ethical_governance.infra.persistence import ReferenceDataManager
from ethical_governance.infra.queue import MessageQueueBase, MockMessageQueue
from ethical_governance.infra.tenancy import CircuitBreaker
from ethical_governance.main import build_demo_dataset, create_app
from ethical_governance.ml.models import ModelRegistry, TrainedModel
from ethical_governance.ml.monitor import ModelMonitor


@pytest.fixture(scope="session")
def demo_df():
    return build_demo_dataset(n=400)


@pytest.fixture(scope="session")
def trained_lr_model(demo_df):
    from sklearn.model_selection import train_test_split
    features = ["income", "savings", "gender"]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=42))])
    pipe.fit(demo_df[features], demo_df["approved"])
    return TrainedModel(
        name="test_model", pipeline=pipe, features=features,
        protected="gender", target="approved", dataset_name="test_credit",
        trained_at="2025-01-01T00:00:00+00:00",
        train_metrics={"accuracy": 0.95, "recall": 0.93, "precision": 0.94},
        reference_meta={"column_stats": {
            "income":  {"mean": 3000.0, "std": 1000.0, "min": 500.0,  "max": 6000.0},
            "savings": {"mean": 10000.0,"std": 5000.0, "min":-5000.0, "max": 30000.0},
            "gender":  {"mean": 0.5,    "std": 0.5,   "min": 0.0,    "max": 1.0},
        }},
    )


@pytest.fixture
def tmp(tmp_path): return tmp_path

@pytest.fixture
def audit(tmp): return AuditLogger(tmp / "audit.jsonl")

@pytest.fixture
def feedback_store(tmp): return FeedbackStore(tmp / "feedback.jsonl")

@pytest.fixture
def monitor(tmp): return ModelMonitor(tmp / "monitor")

@pytest.fixture
def registry(tmp):
    (tmp / "data_ref").mkdir(); (tmp / "models").mkdir()
    return ModelRegistry(tmp / "models", ReferenceDataManager(tmp / "data_ref"))


@pytest.mark.unit
class TestObservabilityService:
    def test_default_is_mock_tracer(self):
        assert isinstance(observability.get_tracer(), MockTracer)

    def test_set_tracer_replaces_instance(self):
        new_tracer = MockTracer()
        observability.set_tracer(new_tracer)
        assert observability.get_tracer() is new_tracer

    def test_span_context_manager_no_crash(self):
        with observability.get_tracer().start_as_current_span("test") as span:
            span.set_attribute("k", "v")

    def test_otel_tracer_class_available(self):
        from ethical_governance.infra.observability import TracerBase
        assert issubclass(OTelTracer, TracerBase)
        assert issubclass(MockTracer, TracerBase)
        assert isinstance(OTelTracer(MockTracer()), TracerBase)

    def test_logger_no_duplicate_handlers(self):
        from ethical_governance.infra.observability import get_logger
        l1 = get_logger("dedup.test"); l2 = get_logger("dedup.test")
        assert l1 is l2 and len(l1.handlers) == 1


@pytest.mark.unit
class TestMessageQueueInterface:
    def test_is_abstract(self):
        from abc import ABC
        assert issubclass(MessageQueueBase, ABC)
        assert "send_message" in MessageQueueBase.__abstractmethods__

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError): MessageQueueBase()

    def test_mock_is_concrete(self):
        assert isinstance(MockMessageQueue(max_retries=2), MessageQueueBase)

    @pytest.mark.asyncio
    async def test_send_succeeds(self):
        q = MockMessageQueue(max_retries=3)
        with patch("numpy.random.rand", return_value=0.99):
            await q.send_message("TOPIC", {"event": "test"})
        assert len(q._sent) == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raise(self):
        q = MockMessageQueue(max_retries=2)
        with patch("numpy.random.rand", return_value=0.01):
            with pytest.raises(RuntimeError, match="Impossibile"):
                await q.send_message("FAIL", {})

    def test_custom_implementation_accepted(self):
        class CustomQ(MessageQueueBase):
            async def send_message(self, topic, message): pass
        assert isinstance(CustomQ(), MessageQueueBase)


@pytest.mark.unit
class TestFairnessAnalyzer:
    def _biased(self, n=200, pr0=0.3, pr1=0.7, seed=42):
        rng = np.random.default_rng(seed); half = n // 2
        g = np.concatenate([np.zeros(half), np.ones(half)]).astype(int)
        a = np.concatenate([rng.binomial(1,pr0,half), rng.binomial(1,pr1,half)])
        return pd.DataFrame({"income":rng.normal(3000,500,n),"savings":rng.normal(10000,2000,n),"gender":g,"approved":a})

    def test_all_five_metrics_present(self, trained_lr_model, demo_df):
        r = FairnessAnalyzer().analyze(trained_lr_model, demo_df, 0.10)
        for attr in ("demographic_parity_difference","demographic_parity_ratio","equal_opportunity_difference","disparate_impact_ratio","predictive_parity_difference"):
            assert isinstance(getattr(r, attr), float)

    def test_severity_monotonic(self, trained_lr_model):
        fa = FairnessAnalyzer()
        r1 = fa.analyze(trained_lr_model, self._biased(pr0=0.48,pr1=0.52), 0.10)
        r2 = fa.analyze(trained_lr_model, self._biased(pr0=0.05,pr1=0.95,n=400), 0.10)
        assert len(r2.violations) >= len(r1.violations)

    def test_bootstrap_ci_ordered(self, trained_lr_model, demo_df):
        ci = FairnessAnalyzer().analyze(trained_lr_model, demo_df, 0.10).bootstrap_ci
        assert ci["ci_2_5"][0] <= ci["mean_dpd"][0] <= ci["ci_97_5"][0]

    def test_empty_group_no_crash(self, trained_lr_model):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"income":rng.normal(3000,500,50),"savings":rng.normal(10000,2000,50),"gender":np.ones(50,dtype=int),"approved":rng.integers(0,2,50)})
        r = FairnessAnalyzer().analyze(trained_lr_model, df, 0.10)
        assert r.n_group_0 == 0


@pytest.mark.unit
class TestDataQualityChecker:
    @pytest.fixture
    def checker(self): return DataQualityChecker(z_threshold=3.0)
    @pytest.fixture
    def ref(self): return {"column_stats":{"income":{"mean":3000.0,"std":500.0,"min":1000.0,"max":5000.0},"savings":{"mean":10000.0,"std":2000.0,"min":2000.0,"max":20000.0}}}

    def test_clean_passes(self, checker, ref):
        assert checker.check(pd.DataFrame([{"income":3200.0,"savings":11000.0}]),ref,"m").passed

    def test_nan_flagged(self, checker, ref):
        r = checker.check(pd.DataFrame([{"income":float("nan"),"savings":11000.0}]),ref,"m")
        assert not r.passed and any("mancante" in i for i in r.issues)

    def test_zscore_outlier(self, checker, ref):
        r = checker.check(pd.DataFrame([{"income":6500.0,"savings":11000.0}]),ref,"m")
        assert not r.passed and any("Z-score" in i for i in r.issues)

    def test_no_ref_passes(self, checker):
        assert checker.check(pd.DataFrame([{"income":99999.0}]),None,"m").passed


@pytest.mark.unit
class TestWeightedRiskEngine:
    def _report(self, dpd=0.0, severity="LOW", violations=None):
        from ethical_governance.core.fairness import FairnessReport
        return FairnessReport(
            model_key="t", protected_attribute="g", group_0_label="g0", group_1_label="g1",
            n_group_0=100, n_group_1=100, demographic_parity_difference=dpd,
            demographic_parity_ratio=max(0.0,1.0-abs(dpd)), equal_opportunity_difference=0.0,
            disparate_impact_ratio=0.9, predictive_parity_difference=0.0, overall_accuracy=0.85,
            accuracy_group_0=0.84, accuracy_group_1=0.86, severity=severity, violations=violations or [],
            bootstrap_ci={"mean_dpd":[abs(dpd)],"ci_2_5":[0.0],"ci_97_5":[abs(dpd)*2]},
            timestamp="2025-01-01T00:00:00+00:00",
        )

    @pytest.mark.asyncio
    async def test_zero_risk(self):
        r = await RiskEngine(CircuitBreaker()).evaluate("t", None, {})
        assert r["risk_score"] == 0.0 and r["risk_level"] == "LOW"

    @pytest.mark.asyncio
    async def test_components_and_weights(self):
        r = await RiskEngine(CircuitBreaker()).evaluate("t", self._report(dpd=0.3,severity="HIGH",violations=["DPD"]), {})
        assert all(k in r["components"] for k in ("fairness","drift","quality"))
        assert abs(sum(r["weights"].values()) - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_circuit_breaker_forces_unacceptable(self):
        cb = CircuitBreaker(); await cb.trip("bad")
        r = await RiskEngine(cb).evaluate("bad", None, {})
        assert r["risk_score"] == 1.0 and r["risk_level"] == "UNACCEPTABLE"
        await cb.reset("bad")

    @pytest.mark.asyncio
    async def test_score_bounded(self):
        bad = self._report(dpd=0.9, severity="CRITICAL", violations=["DPD","DPR","EOD","DIR","PPD"])
        from ethical_governance.core.quality import QualityReport
        r = await RiskEngine(CircuitBreaker()).evaluate(
            "t", bad, {f"f{i}":{"drift_detected":True} for i in range(20)},
            quality=QualityReport(passed=False, issues=["x"]*5, quality_score=0.0, checked_at="now"),
        )
        assert 0.0 <= r["risk_score"] <= 1.0


@pytest.mark.unit
class TestDriftDetector:
    @pytest.fixture
    def det(self): return DriftDetector(p_threshold=0.05)

    def test_numeric_ks(self, det):
        ref = pd.DataFrame({"x": np.random.default_rng(0).normal(3000,500,200)})
        cur = pd.DataFrame({"x": np.random.default_rng(1).normal(3000,500,200)})
        assert det.detect(ref, cur)["x"]["test"] == "ks"

    def test_categorical_chi2(self, det):
        ref = pd.DataFrame({"c": pd.Categorical(["A","B","C","A","B"]*40)})
        cur = pd.DataFrame({"c": pd.Categorical(["A","B","C","A","B"]*40)})
        assert det.detect(ref, cur)["c"]["test"] == "chi2"

    def test_numeric_drift_detected(self, det):
        ref = pd.DataFrame({"x": np.random.default_rng(0).normal(3000,200,400)})
        cur = pd.DataFrame({"x": np.random.default_rng(1).normal(6000,200,400)})
        assert det.detect(ref, cur)["x"]["drift_detected"]

    def test_categorical_drift_detected(self, det):
        ref = pd.DataFrame({"r": pd.Series(["A"]*100+["B"]*100+["C"]*100+["D"]*100, dtype=object)})
        cur = pd.DataFrame({"r": pd.Series(["A"]*390+["B"]*3  +["C"]*4  +["D"]*3,   dtype=object)})
        assert det.detect(ref, cur)["r"]["drift_detected"]

    def test_missing_col_skipped(self, det):
        ref = pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})
        cur = pd.DataFrame({"a":[1,2,3]})
        r = det.detect(ref, cur)
        assert "a" in r and "b" not in r

    def test_bool_is_numeric(self, det):
        ref = pd.DataFrame({"f": pd.array([True,False]*100, dtype=bool)})
        cur = pd.DataFrame({"f": pd.array([True,False]*100, dtype=bool)})
        assert det.detect(ref, cur)["f"]["detected_type"] == "numeric"


@pytest.mark.unit
class TestAlertService:
    def _fairness(self, dpd=0.20, severity="HIGH"):
        from ethical_governance.core.fairness import FairnessReport
        return FairnessReport(
            model_key="m", protected_attribute="g", group_0_label="g0", group_1_label="g1",
            n_group_0=100, n_group_1=100, demographic_parity_difference=dpd,
            demographic_parity_ratio=0.5, equal_opportunity_difference=0.0, disparate_impact_ratio=0.9,
            predictive_parity_difference=0.0, overall_accuracy=0.85, accuracy_group_0=0.84, accuracy_group_1=0.86,
            severity=severity, violations=["DPD"],
            bootstrap_ci={"mean_dpd":[abs(dpd)],"ci_2_5":[0.0],"ci_97_5":[abs(dpd)*2]},
            timestamp="2025-01-01T00:00:00+00:00",
        )

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self, audit):
        svc = AlertService(WebhookNotifier(None), audit)
        assert await svc.evaluate_and_fire("m","t",self._fairness(dpd=0.05),{},0.10) == []

    @pytest.mark.asyncio
    async def test_dpd_alert_fires(self, audit):
        svc = AlertService(WebhookNotifier(None), audit)
        assert len(await svc.evaluate_and_fire("m","t",self._fairness(dpd=0.20),{},0.10)) >= 1

    @pytest.mark.asyncio
    async def test_webhook_called(self, audit):
        notifier = WebhookNotifier("https://example.com/hook")
        svc = AlertService(notifier, audit)
        with patch.object(notifier, "send", new=AsyncMock(return_value=True)) as mock_send:
            await svc.evaluate_and_fire("m","t",self._fairness(dpd=0.25),{},0.10)
        assert mock_send.called

    @pytest.mark.asyncio
    async def test_webhook_failure_no_raise(self, audit):
        svc = AlertService(WebhookNotifier("https://unreachable.invalid", timeout=0.1), audit)
        assert isinstance(await svc.evaluate_and_fire("m","t",self._fairness(dpd=0.25),{},0.10), list)


@pytest.mark.integration
class TestModelCaching:
    @pytest.mark.asyncio
    async def test_same_object_from_cache(self, registry, demo_df):
        m = await registry.train(demo_df,"logistic_regression","credit","gender","approved")
        assert await registry.load(m.name) is await registry.load(m.name)

    @pytest.mark.asyncio
    async def test_clear_single(self, registry, demo_df):
        m = await registry.train(demo_df,"logistic_regression","credit","gender","approved")
        result = await registry.clear_cache(m.name)
        assert result["total"] == 1 and m.name not in registry._cache

    @pytest.mark.asyncio
    async def test_clear_all(self, registry, demo_df):
        await registry.train(demo_df,"logistic_regression","credit","gender","approved")
        await registry.clear_cache()
        assert len(registry._cache) == 0

    @pytest.mark.asyncio
    async def test_load_nonexistent_raises_domain_error(self, registry):
        from ethical_governance.infra.exceptions import DomainError
        with pytest.raises(DomainError):
            await registry.load("ghost_model")


@pytest.mark.integration
class TestFeedbackSecurity:
    @pytest.mark.asyncio
    async def test_find_prediction_by_id(self, audit):
        pid = str(uuid.uuid4())
        await audit.log("AUTO_DECISION", {"model_name":"m","prediction_id":pid,"prediction":1,"tenant_id":"ta"})
        assert (await audit.find_prediction(pid))["tenant_id"] == "ta"

    @pytest.mark.asyncio
    async def test_unknown_prediction_returns_none(self, audit):
        assert await audit.find_prediction(str(uuid.uuid4())) is None

    @pytest.mark.asyncio
    async def test_real_accuracy_correct(self, audit, feedback_store, monitor):
        mn = "acc_test"
        for pid, pred, real in [("p1",1,1),("p2",0,0),("p3",1,1),("p4",1,0),("p5",0,0),("p6",1,1)]:
            await audit.log("AUTO_DECISION", {"model_name":mn,"prediction_id":pid,"prediction":pred,"tenant_id":"t"})
            await feedback_store.store(pid, mn, real, "t")
        r = await monitor.compute_real_accuracy(mn, audit, feedback_store)
        assert r["status"] == "OK" and abs(r["real_accuracy"] - 5/6) < 0.01


@pytest.mark.integration
class TestAutoCorrectionService:
    async def _add(self, audit, model_name, events):
        for ev in events:
            await audit.log("HUMAN_RESOLUTION", {"model_key": model_name, **ev})

    @pytest.mark.asyncio
    async def test_insufficient_data(self, audit):
        svc = AutoCorrectionService(audit)
        await self._add(audit, "m", [{"resolution":"overridden","protected_value":0}]*3)
        assert (await svc.generate_report("m"))["status"] == "INSUFFICIENT_DATA"

    @pytest.mark.asyncio
    async def test_reweigh_above_threshold(self, tmp):
        a = AuditLogger(tmp/"ac.jsonl"); svc = AutoCorrectionService(a)
        events = (
            [{"resolution":"overridden","protected_value":0}]*10 +
            [{"resolution":"approved",  "protected_value":0}]*2  +
            [{"resolution":"overridden","protected_value":1}]*2  +
            [{"resolution":"approved",  "protected_value":1}]*10
        )
        await self._add(a, "m_r", events)
        r = await svc.generate_report("m_r")
        assert r["status"] == "OK" and r["suggested_action"] in ("REWEIGH_AND_RETRAIN","RETRAIN_FULL")


@pytest.mark.integration
class TestRequestIdPropagation:
    @pytest.mark.asyncio
    async def test_request_id_in_log(self, audit):
        pid = str(uuid.uuid4()); rid = str(uuid.uuid4())
        await audit.log("AUTO_DECISION", {"model_name":"m","prediction_id":pid,"prediction":1,"tenant_id":"t","request_id":rid,"latency_ms":12.5,"model_version":"2025-01-01T00:00:00+00:00"})
        event = await audit.find_prediction(pid)
        assert event is not None and event.get("request_id") == rid

    @pytest.mark.asyncio
    async def test_monitor_stores_request_id(self, monitor):
        rid = str(uuid.uuid4())
        await monitor.record("test_model", accuracy=0.92, dpd=0.08, request_id=rid)
        history = await monitor.get_history("test_model")
        assert len(history) == 1 and history[0]["request_id"] == rid

    @pytest.mark.asyncio
    async def test_monitor_default_none(self, monitor):
        await monitor.record("test_model_2", accuracy=0.90, dpd=0.05)
        history = await monitor.get_history("test_model_2")
        assert "request_id" in history[0]


@pytest.mark.api
class TestAPIEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        with TestClient(create_app(), raise_server_exceptions=True) as c:
            yield c

    @pytest.fixture
    def auth(self): return {"X-Tenant-Id":"standard_demo","X-API-Key":"standard-key-456"}
    @pytest.fixture
    def admin(self): return {"X-Admin-Key":"test-admin-key"}

    def test_health_version(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200 and r.json()["version"] == "2.7.0"

    def test_model_not_found_structured_error(self, client, auth):
        r = client.post("/v1/predict", json={"model_name":"nonexistent","features":{"income":1.0}}, headers=auth)
        assert r.status_code == 404 and r.json()["detail"]["error"] == "ModelNotFoundError"

    def test_feature_missing_structured_error(self, client, auth):
        r = client.post("/v1/predict", json={"model_name":"credit_model_v1","features":{"income":3500.0}}, headers=auth)
        assert r.status_code == 400 and r.json()["detail"]["error"] == "FeatureMissingError"

    def test_prediction_not_found_structured_error(self, client, auth):
        r = client.post("/v1/feedback", json={"prediction_id":str(uuid.uuid4()),"model_name":"credit_model_v1","real_outcome":1}, headers=auth)
        assert r.status_code == 404 and r.json()["detail"]["error"] == "PredictionNotFoundError"

    def test_predict_has_forensic_fields(self, client, auth):
        r = client.post("/v1/predict", json={"model_name":"credit_model_v1","features":{"income":3500.0,"savings":12000.0,"gender":1.0}}, headers=auth)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("COMPLETED","PENDING_APPROVAL") and "request_id" in body

    def test_invalid_tenant_403(self, client):
        r = client.post("/v1/predict", json={"model_name":"credit_model_v1","features":{"income":3500.0,"savings":12000.0,"gender":1.0}}, headers={"X-Tenant-Id":"ghost","X-API-Key":"wrong"})
        assert r.status_code == 403

    def test_path_traversal_422(self, client, auth):
        r = client.post("/v1/predict", json={"model_name":"../etc/passwd","features":{"income":1.0}}, headers=auth)
        assert r.status_code == 422

    def test_feedback_invalid_uuid_422(self, client, auth):
        r = client.post("/v1/feedback", json={"prediction_id":"bad","model_name":"m","real_outcome":1}, headers=auth)
        assert r.status_code == 422

    def test_batch_predict(self, client, auth):
        items = [{"income":3000.0+i*100,"savings":10000.0,"gender":float(i%2)} for i in range(3)]
        r = client.post("/v1/predict/batch", json={"model_name":"credit_model_v1","items":items}, headers=auth)
        assert r.status_code == 200 and r.json()["total"] == 3

    def test_batch_too_large_422(self, client, auth):
        items = [{"income":float(i),"savings":1.0,"gender":0.0} for i in range(20)]
        r = client.post("/v1/predict/batch", json={"model_name":"credit_model_v1","items":items}, headers=auth)
        assert r.status_code == 422

    def test_compare_endpoint(self, client, auth):
        r = client.get("/v1/compare?models=credit_model_v1,credit_model_v1&sample_size=100", headers=auth)
        assert r.status_code == 200 and "recommended_model" in r.json()

    def test_compare_single_model_400(self, client, auth):
        assert client.get("/v1/compare?models=credit_model_v1", headers=auth).status_code == 400

    def test_retrain_trigger(self, client, auth):
        r = client.post("/v1/retrain/trigger", json={"model_name":"credit_model_v1"}, headers=auth)
        assert r.status_code == 200 and r.json()["status"] in ("SKIPPED","COMPLETED","ERROR")

    def test_quality_check(self, client, auth):
        r = client.post("/v1/quality-check", json={"model_name":"credit_model_v1","features":{"income":3500.0,"savings":12000.0,"gender":1.0}}, headers=auth)
        assert r.status_code == 200 and "passed" in r.json()

    def test_audit_endpoint(self, client, auth):
        assert client.get("/v1/audit?n=5", headers=auth).status_code == 200

    def test_metrics_prometheus(self, client):
        r = client.get("/v1/metrics")
        assert r.status_code == 200 and "gov_" in r.text

    def test_cache_clear_admin(self, client, admin):
        r = client.post("/v1/admin/cache/clear", headers=admin)
        assert r.status_code == 200 and "cleared" in r.json()

    def test_cache_clear_requires_admin(self, client, auth):
        assert client.post("/v1/admin/cache/clear", headers=auth).status_code == 422

    def test_circuit_breaker(self, client, admin):
        r = client.post("/v1/admin/circuit-breaker/free_demo", json={"action":"trip"}, headers=admin)
        assert r.status_code == 200
        client.post("/v1/admin/circuit-breaker/free_demo", json={"action":"reset"}, headers=admin)

    def test_corrections_endpoint(self, client, auth):
        assert client.get("/v1/corrections/credit_model_v1", headers=auth).status_code == 200

    def test_monitor_endpoint(self, client, auth):
        r = client.get("/v1/monitor/credit_model_v1", headers=auth)
        assert r.status_code == 200 and "history" in r.json()
