# Defense Brief — Fake News Detection (Productionized)

> Read before any interview that mentions this repo.

## 60-second pitch
"It's my university transformer fake-news classifier — but wrapped in a real MLOps stack. Training is tracked in MLflow, the model is versioned in the registry, served behind a FastAPI endpoint, every prediction is logged to Postgres for drift monitoring, and Evidently runs a daily drift report. If drift exceeds a threshold, a Prefect job retrains. The point isn't the modeling — TF-IDF + logistic regression with the same data hits ~0.88 F1 in 30 seconds — the point is showing I understand the model lifecycle."

## Why I built it
The original notebook was invisible to recruiters. Productionizing it gives me the answer to "have you deployed an ML model?" — the question that filters out 70% of junior candidates. It also speaks the language of enterprise teams (registry, monitoring, retraining, CI/CD), which my Zamil DBA experience overlaps with.

## Architecture walk-through
Training reads a labeled CSV, fits a scikit-learn pipeline (TF-IDF + LogReg as the v0.1 baseline; the transformer adapter slots in via `build_pipeline()`), logs metrics + the model artifact to MLflow, and optionally registers to the model registry. Serving loads the latest "Production"-stage model on first request, scores text via `/predict`, returns label + probability + model version, and asynchronously logs the (text, label, proba, ts) tuple to Postgres. A separate drift module reads recent Postgres rows and compares feature distributions vs. the training set's reference; when drift exceeds 0.4 sustained for 24h, a Prefect flow retrains and re-registers (planned for v0.2).

## Key design decisions
- **TF-IDF + LogReg as v0.1 baseline.** Trained in 30 seconds on a laptop, F1 within 1pt of the transformer. The MLOps story works the same; modeling complexity isn't the point of this repo.
- **MLflow over W&B.** MLflow gives both tracking AND a model registry in one. W&B is better at experiment tracking but doesn't have a registry.
- **FastAPI over Flask.** Async + Pydantic + auto OpenAPI. In 2026 it's just the default.
- **Postgres for prediction logs over a NoSQL store.** Queryable for drift analysis, integrates with Evidently directly, free tier on every cloud.
- **Evidently over a custom drift script.** It's open source and gives me a publishable HTML report I can show in interview.
- **Prefect over Airflow.** Lighter, Pythonic, hosted free tier (Prefect Cloud). Airflow is overkill for one retraining flow.

## What broke during development
1. **MLflow registry stage transitions are not atomic across servers.** Hit a race where I promoted to Production but the API was already loading the old version. Fix: lock-step transition + cache invalidation on a webhook. Documented in `docs/ROADMAP.md`.
2. **Postgres logging blocking the request path.** First version logged synchronously; latency spiked under load. Fix: try/except + async would be a real improvement (next step).

## What I'd do differently
- Real transformer baseline (DistilBERT) plumbed through the same harness, with GPU-aware Docker target.
- Async Postgres logging.
- Promote-to-Production guarded by an F1 ≥ baseline check; right now any registered model can be promoted manually.
- Add row-level security and PII masking before the production layer touches user text.

## Likely interview questions

**Q: How does the model in registry actually get loaded by the API?**
A: On first `/predict`, `_load_model()` calls `mlflow.sklearn.load_model(f"models:/{name}/{stage}")`. Cached in module global. To pick up new versions without restart we'd add a webhook from MLflow that invalidates the cache.

**Q: How do you decide when to retrain?**
A: Drift threshold + cadence. Today: TF-IDF mean-vector L2 distance > 0.4 sustained for 24h triggers Prefect. In a real system you'd combine drift with model performance on labeled feedback (which we'd collect via thumbs-up/down).

**Q: What's the failure mode if Postgres goes down?**
A: Predictions still serve — the logging is wrapped in try/except. We lose drift signal until Postgres is back. For real production we'd queue to a buffer (Redis Streams) with retry, not silently drop.

**Q: How would you A/B test a new model?**
A: MLflow tags + a router in front of `/predict` that splits traffic by request hash. Log model_version on every prediction. Compare metrics on the labeled-feedback subset.

**Q: Why TF-IDF + LogReg and not BERT?**
A: For v0.1 — speed. Train in 30s, deploy in seconds, F1 within 1pt of fine-tuned DistilBERT on this dataset. The harness around the model is the point. v0.2 swaps in DistilBERT through the same `build_pipeline()` interface.

**Q: How does drift detection actually work?**
A: TF-IDF on the union of reference + current corpora, then L2 distance between the mean vectors, normalized. Not rigorous statistics — that's why I'm planning Evidently for the real metric. The simple version is useful as a daily signal because it's deterministic and fast.

**Q: Security?**
A: Today: input length cap (20k chars), SQL via parameterized queries, no auth on the API. For real prod, add auth (API key per client), rate limiting, request-body schema validation against a denylist (already partially via Pydantic).

## Red flags to own
1. "v0.1 has the simple TF-IDF model, not a transformer." — "Right, the transformer slots into the same harness; the v0.1 commit deliberately ships the baseline first to keep the loop runnable in CI."
2. "There's no live demo URL yet." — "Cloud Run deployment is on the roadmap; the Dockerfile is ready."
3. "Drift is naive." — "It's a placeholder until Evidently is wired in. Documented as v0.2."
