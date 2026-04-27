# Roadmap

## v0.1 (now)
- [x] training script with MLflow tracking + register
- [x] FastAPI /predict with Postgres prediction logging
- [x] drift detector (basic TF-IDF distance)
- [x] Dockerfile + docker-compose stack
- [x] GitHub Actions CI

## v0.2
- [ ] Evidently drift dashboard scheduled hourly
- [ ] Prefect retraining flow when drift > 0.4 sustained 24h
- [ ] Promote-to-Production guarded by F1 ≥ baseline
- [ ] HuggingFace transformer baseline behind same interface
- [ ] /predict_batch endpoint

## v0.3
- [ ] Live demo on GCP Cloud Run
- [ ] Model card on HF Hub
- [ ] Public dataset card with license
