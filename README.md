# Fake News Detection — Productionized

An end-to-end MLOps pipeline wrapped around a fake-news text classifier: training and experiment tracking in MLflow, the model versioned in MLflow's registry, FastAPI model serving behind a `/predict` endpoint, every prediction logged to PostgreSQL, and drift monitored with Evidently AI. Built to demonstrate the lifecycle of an ML model — not just the modeling. The classifier itself is a continuation of a university transformer-based fake-news project (0.89 validation accuracy, separate dataset); the interesting part of this repo is what's around the model, not the model.

## What it does

- Trains a TF-IDF + Logistic Regression baseline classifier, with every run tracked in MLflow (metrics, params, classification report, model artifact) and optionally registered to the MLflow model registry.
- Serves the latest "Production"-stage model from the registry behind a FastAPI `/predict` endpoint with a Pydantic request/response schema.
- Logs every prediction (timestamp, input text, label, probability) to PostgreSQL via SQLAlchemy — the reference data drift monitoring reads from.
- Computes drift two ways: an Evidently AI `DataDriftPreset` + `TextOverviewPreset` HTML report comparing the training reference against the last 7 days of logged Postgres predictions, with a lightweight TF-IDF mean-vector distance score as a fallback if Evidently isn't available.
- Ships a local Docker Compose stack (Postgres + MLflow server) for one-command local infra, plus a Dockerfile for the API itself.
- Has CI (GitHub Actions: lint with ruff, run pytest) and tests for both the serving schema and the drift detector.

## Architecture

```
                         ┌────────────────────┐
   labeled corpus ─────▶ │  training script    │  src/train/train.py
                         │  (MLflow tracking)  │
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
                     │ FastAPI  │ ─┴─▶ │  PostgreSQL  │
                     │  /predict│      │  predictions │
                     └──────────┘      └──────┬───────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │ Evidently AI │  DataDriftPreset +
                                      │ drift report │  TextOverviewPreset
                                      └──────────────┘
```

**Stack:** scikit-learn (TF-IDF + Logistic Regression baseline, swappable for a HuggingFace transformer via the same `build_pipeline()` interface), MLflow for experiment tracking and the model registry, FastAPI + Pydantic for serving, PostgreSQL + SQLAlchemy for prediction logging, Evidently AI for drift reports.

**Planned, not yet in this repo** (see `docs/ROADMAP.md`): DVC-based data versioning, a scheduled Evidently drift job, and a Prefect flow to trigger retraining automatically when drift crosses a threshold — today the drift report and retraining are run manually.

## How to run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start local infra (Postgres + MLflow server)
docker compose -f deploy/docker-compose.yml up -d postgres mlflow

# 3. Train and register a model
python -m src.train.train data/seed_labeled.csv --register

# 4. Serve
PYTHONPATH=. uvicorn src.serve.app:app --reload

# 5. Predict
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text":"Aliens land in Riyadh, govt confirms"}'
```

Equivalent `make` targets: `make install`, `make compose-up`, `make train`, `make serve`, `make test`, `make drift-report`.

Set `DATABASE_URL` (see `.env.example`) for prediction logging and drift reports to read from Postgres; without it, `/predict` still works but skips logging, and `run_evidently.py` falls back to a synthetic sample of the reference data.

## Results

```bash
python -m src.train.train data/seed_labeled.csv --register
```

Held-out F1 on `data/seed_labeled.csv` (30 rows) is **0.000** — the seed set is too small for the TF-IDF + LogReg baseline to learn anything beyond the majority class. This is expected and documented, not a hidden failure: the point of this baseline is to prove the MLflow → FastAPI → Postgres → Evidently wiring works end-to-end, not to hit a high score on a placeholder dataset. Swap in a real corpus (LIAR / FakeNewsNet, per `docs/ROADMAP.md`) to evaluate the modeling layer — the pipeline harness doesn't change.

## License

MIT
