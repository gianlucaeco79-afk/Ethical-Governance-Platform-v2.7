# Ethical Governance Platform v2.7

https://doi.org/10.5281/zenodo.19545702

Piattaforma per la governance etica di modelli ML in produzione.
Monitora bias, drift e rischi in tempo reale con un'architettura modulare pronta per la produzione.

---

## Novità della V2.7

La V2.7 introduce tre miglioramenti architetturali:
architettura modulare, ObservabilityService injectable, MessageQueueBase ABC.

---

## Struttura del pacchetto

```text
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
```

---

## Funzionalità

| Componente | Descrizione |
|---|---|
| **BatchPredictor** | N predizioni in parallelo, partial success |
| **RetrainingPipeline** | Loop feedback → AutoCorrection → retraining |
| **AlertService** | Webhook push su DPD alto, drift, risk score |
| **ModelComparator** | Confronto fairness tra modelli con raccomandazione |
| **FairnessAnalyzer** | DPD, DPR, EOD, DIR, PPD + bootstrap CI |
| **DriftDetector** | KS test (numerico) + Chi² contingency (categorico) |
| **RiskEngine** | Pesi configurabili da .env |
| **Audit Trail forense** | request_id, latency_ms, model_version in ogni log |
| **HITL Manager** | Coda task umani con approve/reject |
| **Prometheus** | 20+ metriche: Counter, Gauge, Histogram |

---

## Installazione

```bash
pip install -r requirements.txt
uvicorn ethical_governance.main:app --reload --port 8000
```

---

## Test

```bash
pytest test_main.py -v
pytest test_main.py -m unit
pytest test_main.py -m integration
pytest test_main.py -m api
```

81 test — layer: unit / integration / api

---


**Autore:** Gianluca Ecora 

**Licenza:** MIT

    
