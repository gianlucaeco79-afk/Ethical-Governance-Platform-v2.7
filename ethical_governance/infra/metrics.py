"""
infra/metrics.py — Tutte le metriche Prometheus in un unico modulo.

Importare le singole metriche dove servono:
    from ethical_governance.infra.metrics import PREDICT_REQUESTS, FAIRNESS_SCORE
"""
from prometheus_client import Counter, Gauge, Histogram

PREDICT_REQUESTS    = Counter("gov_predict_total",           "Predizioni totali",              ["tenant_id", "model_name", "status"])
PREDICT_LATENCY     = Histogram("gov_predict_latency_s",     "Latenza predizioni (s)",          ["model_name"], buckets=[.01,.05,.1,.25,.5,1,2,5])
FAIRNESS_SCORE      = Gauge("gov_fairness_dpd",              "DPD corrente",                   ["model_name"])
DRIFT_DETECTED      = Counter("gov_drift_detected_total",    "Feature con drift",              ["model_name"])
HITL_PENDING        = Gauge("gov_hitl_pending",              "Task HITL in attesa")
HITL_RESOLVED       = Counter("gov_hitl_resolved_total",     "Task HITL risolti",              ["resolution"])
CB_TRIPS            = Counter("gov_cb_trips_total",          "Circuit breaker trip",           ["tenant_id"])
FAIRNESS_VIOLATIONS = Counter("gov_fairness_violations",     "Violazioni fairness",            ["model_name", "metric"])
QUALITY_VIOLATIONS  = Counter("gov_quality_violations",      "Violazioni data quality",        ["model_name", "check"])
MODEL_DRIFT_GAUGE   = Gauge("gov_model_accuracy_drift",      "Drift accuracy nel tempo",       ["model_name"])
AUTO_CORRECTIONS    = Counter("gov_auto_corrections",        "Suggerimenti autocorrection",    ["model_name"])
FEEDBACK_TOTAL      = Counter("gov_feedback_total",          "Feedback ricevuti",              ["model_name"])
REAL_ACCURACY_GAUGE = Gauge("gov_real_accuracy",             "Accuracy reale",                 ["model_name"])
FEEDBACK_REJECTED   = Counter("gov_feedback_rejected_total", "Feedback rifiutati",             ["reason"])
CACHE_HITS          = Counter("gov_cache_hits_total",        "Model cache hits",               ["model_name"])
CACHE_MISSES        = Counter("gov_cache_misses_total",      "Model cache misses",             ["model_name"])
BATCH_REQUESTS      = Counter("gov_batch_requests_total",    "Batch prediction richieste",     ["tenant_id", "model_name"])
BATCH_SIZE_HIST     = Histogram("gov_batch_size",            "Dimensione dei batch",           buckets=[1,5,10,20,30,50])
RETRAIN_TOTAL       = Counter("gov_retrain_total",           "Retraining avviati",             ["model_name", "action"])
ALERTS_FIRED        = Counter("gov_alerts_fired_total",      "Alert inviati",                  ["rule", "severity"])
COMPARE_REQUESTS    = Counter("gov_compare_requests_total",  "Confronti tra modelli",          ["tenant_id"])
