"""FastAPI inference service.

Loads the latest 'Production' model from MLflow registry on startup, scores
text, logs every prediction to Postgres for drift monitoring downstream.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Fake News Detector", version="0.1.0")
_model = None  # lazy-loaded on first request


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class PredictResponse(BaseModel):
    label: int
    label_name: str
    proba: float
    model_version: str


@app.on_event("startup")
def _warm() -> None:
    """Load model lazily on first /predict, but warm here in dev."""
    return None


def _load_model():
    global _model
    if _model is not None:
        return _model
    import mlflow.sklearn  # type: ignore

    name = os.environ.get("MODEL_NAME", "fake-news-clf")
    stage = os.environ.get("MODEL_STAGE", "Production")
    _model = mlflow.sklearn.load_model(f"models:/{name}/{stage}")
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _load_model()
    proba = float(model.predict_proba([req.text])[0, 1])
    label = int(proba >= 0.5)
    _log_prediction(req.text, label, proba)
    return PredictResponse(
        label=label,
        label_name="fake" if label == 1 else "real",
        proba=proba,
        model_version=os.environ.get("MODEL_STAGE", "Production"),
    )


def _log_prediction(text: str, label: int, proba: float) -> None:
    """Log to Postgres if DATABASE_URL is set; otherwise no-op."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        return
    try:
        from sqlalchemy import create_engine, text as sql_text  # type: ignore

        engine = create_engine(url)
        with engine.begin() as conn:
            conn.execute(
                sql_text(
                    "INSERT INTO predictions (ts, input_text, label, proba) "
                    "VALUES (:ts, :t, :l, :p)"
                ),
                {
                    "ts": datetime.now(timezone.utc),
                    "t": text[:2000],
                    "l": label,
                    "p": proba,
                },
            )
    except Exception:  # noqa: BLE001
        # Log to stdout but don't fail the request
        import logging

        logging.exception("prediction logging failed")
