"""Generate an Evidently HTML drift report from logged predictions vs. training reference.

Reads:
  - reference data: `data/seed_labeled.csv` (training set)
  - current data:   recent rows from Postgres (last 7 days of predictions)

Writes:
  - `monitoring/drift_{date}.html`

Usage:
    python -m src.monitor.run_evidently
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def load_reference() -> pd.DataFrame:
    return pd.read_csv("data/seed_labeled.csv")


def load_current_from_postgres(days: int = 7) -> pd.DataFrame:
    """Pull last N days of predictions for drift comparison."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        # Local fallback: synthetic "current" — useful for CI and dry-runs
        ref = load_reference()
        return ref.sample(min(len(ref), 50), random_state=1).reset_index(drop=True)
    from sqlalchemy import create_engine

    engine = create_engine(url)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    df = pd.read_sql(
        "SELECT input_text AS text, label, proba FROM predictions WHERE ts >= %s ORDER BY ts DESC LIMIT 5000",
        engine,
        params=(cutoff,),
    )
    return df


def main() -> None:
    reference = load_reference()
    current = load_current_from_postgres()
    out_dir = Path("monitoring")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d')}.html"

    try:
        from evidently.metric_preset import (  # type: ignore
            DataDriftPreset,
            TextOverviewPreset,
        )
        from evidently.report import Report  # type: ignore

        report = Report(metrics=[DataDriftPreset(), TextOverviewPreset(column_name="text")])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(out_path))
        print(f"Drift report written: {out_path}")
    except Exception as e:  # noqa: BLE001
        print(f"Evidently failed ({e!r}); falling back to lightweight drift score.")
        from .drift import feature_drift

        score = feature_drift(reference["text"].tolist(), current["text"].tolist())
        out_path.write_text(
            f"<html><body><h1>Drift score (TF-IDF baseline)</h1>"
            f"<p>Score: {score:.3f}  (0 = no drift, 1 = max drift)</p></body></html>"
        )
        print(f"Fallback drift score written: {out_path}")


if __name__ == "__main__":
    main()
