import pytest
from fastapi.testclient import TestClient
from server.app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "HFEngine"

def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0

def test_telemetry(client):
    response = client.get("/v1/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "batching" in data
    assert "speculative" in data
    assert "cache_eviction" in data

def test_completions_real_model(client):
    response = client.post(
        "/v1/completions", 
        json={"prompt": "Say hi", "max_tokens": 5, "mode": "real"}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    content = b""
    for chunk in response.iter_bytes():
        content += chunk
    assert len(content) > 0

def test_completions_batching_mode(client):
    response = client.post(
        "/v1/completions", 
        json={"prompt": "Test batching", "max_tokens": 5, "mode": "batching"}
    )
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert len(content) > 0

def test_completions_speculative_mode(client):
    response = client.post(
        "/v1/completions", 
        json={"prompt": "Test speculative", "max_tokens": 5, "mode": "speculative"}
    )
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert len(content) > 0

def test_completions_eviction_mode(client):
    response = client.post(
        "/v1/completions", 
        json={"prompt": "Test cache eviction", "max_tokens": 5, "mode": "eviction"}
    )
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert len(content) > 0

def test_completions_zero_tokens(client):
    response = client.post(
        "/v1/completions", 
        json={"prompt": "Zero test", "max_tokens": 0, "mode": "real"}
    )
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert len(content) == 0
