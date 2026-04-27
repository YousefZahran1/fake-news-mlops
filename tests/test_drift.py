from src.monitor.drift import feature_drift


def test_no_drift_on_identical_corpora():
    a = ["the quick brown fox", "jumps over the lazy dog", "lorem ipsum"]
    assert feature_drift(a, a) < 0.05


def test_high_drift_on_disjoint_corpora():
    a = ["health insurance copay outpatient"] * 5
    b = ["alien spaceship landed roswell"] * 5
    assert feature_drift(a, b) > 0.3


def test_handles_empty():
    assert feature_drift([], ["x"]) == 0.0
    assert feature_drift(["x"], []) == 0.0
