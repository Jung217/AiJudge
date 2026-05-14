"""FastAPI service smoke tests.

Skipped automatically if the model bundle hasn't been trained yet — the bundle
is gitignored (~70MB pickle), so CI checkouts won't have one. Run locally
after `python scripts/04_train_baseline.py --save data/processed/baseline_model.pkl`.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BUNDLE = Path("data/processed/baseline_model.pkl")
pytestmark = pytest.mark.skipif(
    not BUNDLE.exists(),
    reason="model bundle not trained yet — run scripts/04_train_baseline.py --save",
)


@pytest.fixture(scope="module")
def client():
    from app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["bundle_loaded"] is True


def test_version_includes_metadata(client):
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["feature_count"] > 0
    md = body["metadata"]
    # Trained baseline always sets these:
    assert "delta25" in md and "delta75" in md
    assert "probation_threshold" in md


def test_predict_returns_quantile_band_and_constraint(client):
    """Typical 施用第二級毒品 simplified judgment — sentence should be short."""
    payload = {
        "behaviors": ["施用"],
        "drug_levels": [2],
        "can_convert_to_fine": True,
        "jcase": "簡",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    pred = body["prediction"]
    assert pred["p50_months"] > 0
    assert pred["p25_months"] <= pred["p50_months"] <= pred["p75_months"]
    assert body["constraint"] is not None
    assert body["disclaimer"].startswith("本模型")


def test_predict_rule_clip_pushes_into_statutory_range(client):
    """販賣第一級 with no reduction → floor is enormous; the rule clip MUST
    pull any raw prediction up to the floor."""
    payload = {
        "behaviors": ["販賣"], "drug_levels": [1],
    }
    r = client.post("/predict", json=payload)
    body = r.json()
    pred = body["prediction"]
    cs = body["constraint"]
    assert pred["rule_applied"] is True
    assert pred["p50_months"] >= cs["min_months"] - 1e-6
    assert pred["p50_months"] <= cs["max_months"] + 1e-6


def test_predict_aggregate_uses_only_30y_ceiling(client):
    payload = {
        "behaviors": ["施用", "持有"], "drug_levels": [1, 2],
        "is_aggregate_sentence": True, "n_sentence_counts": 2,
    }
    r = client.post("/predict", json=payload)
    body = r.json()
    cs = body["constraint"]
    assert cs["min_months"] == 0
    assert cs["max_months"] == 360  # §51 三十年上限


def test_predict_rejects_empty_behaviors(client):
    r = client.post("/predict", json={"behaviors": [], "drug_levels": [1]})
    assert r.status_code == 422  # pydantic min_length=1


def test_predict_response_always_carries_disclaimer(client):
    r = client.post("/predict", json={"behaviors": ["施用"], "drug_levels": [2]})
    body = r.json()
    assert "disclaimer" in body
    assert "不可" in body["disclaimer"]
    assert "法律建議" in body["disclaimer"]
