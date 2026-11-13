from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "embedding_backend" in body
    assert "reranker_backend" in body


def test_resolve_ticket_returns_structured_response():
    response = client.post(
        "/v1/tickets", json={"query": "I was charged twice for my subscription"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "billing_issue"
    assert body["priority"] == "high"
    assert body["escalation_required"] is True
    assert isinstance(body["sources"], list)


def test_resolve_ticket_rejects_empty_query():
    response = client.post("/v1/tickets", json={"query": "   "})
    assert response.status_code == 422
