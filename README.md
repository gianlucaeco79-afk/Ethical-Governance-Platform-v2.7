Aegis Core v2.7 — Ethical Governance Platform
Piattaforma per la governance etica di modelli ML in produzione.
Monitora bias, drift e rischi in tempo reale con un'architettura modulare pronta per la produzione.

Novità della V2.7
La V2.7 introduce tre miglioramenti architetturali ispirati ai pattern di V15.5:
architettura modulare, ObservabilityService injectable, MessageQueueBase ABC.
Struttura del pacchetto

ethical_governance/
├── config.py
├── main.py
├── infra/
│   ├── observability.py
│   ├── metrics.py
│   ├── queue.py
│   ├── audit.py
│   ├── persistence.py
│   ├── tenancy.py
│   ├── alerts.py
│   └── exceptions.py
├── core/
│   ├── quality.py
│   ├── fairness.py
│   ├── drift.py
│   └── risk.py
├── ml/
│   ├── models.py
│   ├── explainability.py
│   ├── monitor.py
│   └── retraining.py
├── governance/
│   ├── hitl.py
│   ├── correction.py
│   ├── batch.py
│   ├── comparison.py
│   └── engine.py
└── api/
    ├── schemas.py
    ├── dependencies.py
    └── routes.py

    Funzionalità

ComponenteDescrizioneBatchPredictorN predizioni in parallelo, partial successRetrainingPipelineLoop feedback → AutoCorrection → retrainingAlertServiceWebhook push su DPD alto, drift, risk scoreModelComparatorConfronto fairness tra modelli con raccomandazioneFairnessAnalyzerDPD, DPR, EOD, DIR, PPD + bootstrap CI

DriftDetectorKS test (numerico) + Chi² contingency (categorico)RiskEnginePesi configurabili da .envAudit Trail forenserequest_id, latency_ms, model_version in ogni logHITL ManagerCoda task umani con approve/rejectPrometheus20+ metriche: Counter, Gauge, Histogram

pip install -r requirements.txt
uvicorn ethical_governance.main:app --reload --port 8000

pytest test_main.py -v
pytest test_main.py -m unit
pytest test_main.py -m integration
pytest test_main.py -m api

81 test — layer: unit / integration / api

Autore: Gianluca Ecora
Licenza: MIT
