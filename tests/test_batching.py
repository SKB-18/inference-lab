import pytest
from fastapi.testclient import TestClient
from server.app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_batching_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_batching_completions(client):
    response = client.post(
        "/v1/completions",
        json={"prompt": "Test batching", "max_tokens": 10, "mode": "batching"},
    )
    assert response.status_code == 200
    content = response.content.decode()
    assert len(content) > 0
