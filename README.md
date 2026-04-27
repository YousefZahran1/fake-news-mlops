# Fake News Detection — Productionized

A transformer fake-news classifier wrapped in a real MLOps loop: training tracked in MLflow, model versioned in the registry, served behind a FastAPI endpoint, predictions logged to Postgres, drift monitored with Evidently, retraining triggered when drift exceeds threshold.

> Built to demonstrate the lifecycle of an ML model — not just the modeling. The classifier itself is a continuation of my university transformer-based fake-news project (validation 0.89). The interesting part of this repo is what's around the model, not the model.

## Architecture

```
                         ┌────────────────────┐
   labeled corpus ─────▶ │ training pipeline  │
                         │  (DVC + MLflow)    │
                         └─────────┬──────────┘
                                   │ register best model
                                   ▼
                         ┌────────────────────┐
                         │  Model Registry    │
                         │     (MLflow)       │
                         └─────────┬──────────┘
                                   │ load latest "Production"
                                   ▼
   user request ───▶ ┌──────────┐  │   ┌──────────────┐
                     │ FastAPI  │ ─┴─▶ │  Postgres    │
                     │  /predict│      │  predictions │
                     └──────────┘      └──────┬───────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │ Evidently    │
                                      │ drift report │
                                      └──────┬───────┘
                                             │ if drift > τ
                                             ▼
                                      ┌──────────────┐
                                      │ retrain      │
                                      │ (Prefect job)│
                                      └──────────────┘
```

## Tech stack and why

| Layer | Pick | Why |
|---|---|---|
| Training | scikit-learn + transformers | Same stack as the original project; nothing exotic |
| Tracking | MLflow | Industry-standard; registry + tracking in one |
| Data versioning | DVC | Lightweight, git-friendly; alternative to LakeFS |
| Serving | FastAPI | Async, types, auto OpenAPI |
| Logging | Postgres + SQLAlchemy | Free-tier friendly; queryable |
| Drift | Evidently AI | Open source, decent reports out of the box |
| Scheduling | Prefect (or GH Actions cron) | Prefect for real prod, GH Actions for the demo version |

## Quick start

```bash
pip install -r requirements.txt
docker compose -f deploy/docker-compose.yml up -d postgres mlflow
python -m src.train.train data/labeled.csv
PYTHONPATH=. uvicorn src.serve.app:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"Aliens land in Riyadh, govt confirms"}'
```

## What's done in v0.1 (this commit set)
- [x] training script with MLflow tracking
- [x] model registry contract (load_latest_production)
- [x] FastAPI inference endpoint with Pydantic schema
- [x] Postgres logging of every prediction
- [x] Dockerfile + docker-compose for local stack
- [x] GitHub Actions CI: lint + tests
- [x] Tests for serving contract + drift detector

## What's planned (see ROADMAP.md)
- [ ] Evidently drift report scheduled
- [ ] Prefect retraining flow
- [ ] Live demo on GCP Cloud Run

## License
MIT
