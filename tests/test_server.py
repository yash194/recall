"""Tests for the FastAPI server."""
from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
    from recall.server import app
except ImportError:
    pytest.skip("fastapi not installed", allow_module_level=True)


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "Recall"


def test_observe_then_recall(client):
    # observe
    r = client.post("/v1/memory/observe", json={
        "tenant": "srv_test",
        "user_msg": "Decision: we use Redis Streams.",
        "agent_msg": "OK.",
        "scope": {"p": "1"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["drawer_id"] is not None

    # recall
    r = client.post("/v1/memory/recall", json={
        "tenant": "srv_test",
        "query": "queue",
        "scope": {"p": "1"},
        "mode": "symmetric",
    })
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body
    assert "edges" in body


def test_stats_and_consolidate(client):
    client.post("/v1/memory/observe", json={
        "tenant": "srv_consolidate",
        "user_msg": "We tried LISTEN/NOTIFY but it failed.",
        "agent_msg": "Yes.",
    })
    r = client.get("/v1/memory/stats?tenant=srv_consolidate")
    assert r.status_code == 200
    assert "active_nodes" in r.json()

    r = client.post("/v1/memory/consolidate?tenant=srv_consolidate&budget=5")
    assert r.status_code == 200
    body = r.json()
    assert "regions_processed" in body


def test_audit_endpoint(client):
    client.post("/v1/memory/observe", json={
        "tenant": "srv_audit",
        "user_msg": "Some real fact.",
        "agent_msg": "OK.",
    })
    r = client.get("/v1/memory/audit?tenant=srv_audit")
    assert r.status_code == 200
    assert "entries" in r.json()
