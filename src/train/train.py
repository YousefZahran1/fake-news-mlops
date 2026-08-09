"""Train a fake-news classifier and register it with MLflow.

Lightweight baseline on TF-IDF + Logistic Regression so this repo runs without
a GPU. The university transformer-based version drops in by swapping
`build_pipeline()` to a HF model adapter — the rest of the harness is unchanged.

Usage:
    python -m src.train.train data/labeled.csv --register
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_csv", type=Path, help="CSV with columns: text, label")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "fake-news-clf"))
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    if "text" not in df or "label" not in df:
        raise SystemExit("CSV must have columns: text, label")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str), df["label"].astype(int), test_size=args.test_size, random_state=42
    )

    mlflow.set_experiment("fake-news")
    with mlflow.start_run() as run:
        pipe = build_pipeline()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1 = f1_score(y_test, preds)
        mlflow.log_metric("f1", f1)
        mlflow.log_param("model", "tfidf+logreg")
        report = classification_report(y_test, preds)
        mlflow.log_text(report, "classification_report.txt")
        print(report)

        artifact_path = "model"
        mlflow.sklearn.log_model(pipe, artifact_path=artifact_path)
        if args.register:
            uri = f"runs:/{run.info.run_id}/{artifact_path}"
            mlflow.register_model(uri, args.model_name)
        print(f"f1={f1:.3f}  run_id={run.info.run_id}")


if __name__ == "__main__":
    main()
