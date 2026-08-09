"""Drift detector — computes basic distributional drift between two text
corpora using token-distribution KS-test on TF-IDF features.

Uses Evidently for the rich report; this module is the lightweight contract.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def feature_drift(reference: Iterable[str], current: Iterable[str]) -> float:
    """Return a simple drift score in [0,1] — higher = more drifted.

    Method: TF-IDF on the union, take L2 distance between mean vectors,
    normalize. Not statistically rigorous, but useful as a daily signal.
    For real production, swap with `evidently.metrics.DataDrift`.
    """
    ref = list(reference)
    cur = list(current)
    if not ref or not cur:
        return 0.0
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=1)
    vec.fit(ref + cur)
    R = vec.transform(ref).mean(axis=0)
    C = vec.transform(cur).mean(axis=0)
    # both are matrices; convert to dense vector
    diff = np.asarray(R - C).ravel()
    base = np.linalg.norm(np.asarray(R).ravel()) + 1e-9
    drift = float(np.linalg.norm(diff) / base)
    return min(drift, 1.0)
