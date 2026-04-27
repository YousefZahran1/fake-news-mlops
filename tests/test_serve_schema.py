"""Schema-level test for the FastAPI app — does not load a real model."""
from fastapi.testclient import TestClient

from src.serve.app import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_validation():
    client = TestClient(app)
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422  # min_length=1
