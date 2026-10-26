import pytest
from fastapi.testclient import TestClient
from server.app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_empty_prompt(client):
    response = client.post("/v1/completions", json={"prompt": "", "max_tokens": 5})
    assert response.status_code == 200

def test_zero_max_tokens(client):
    response = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 0})
    assert response.status_code == 200
    assert len(response.content) == 0

def test_extreme_temperature_greedy(client):
    response = client.post(
        "/v1/completions", 
        json={"prompt": "1 2 3", "max_tokens": 3, "temperature": 0.0}
    )
    assert response.status_code == 200
    assert len(response.content) > 0

def test_extreme_temperature_high(client):
    response = client.post(
        "/v1/completions", 
        json={"prompt": "1 2 3", "max_tokens": 3, "temperature": 1.5, "top_p": 0.95}
    )
    assert response.status_code == 200
    assert len(response.content) > 0

def test_missing_prompt_field(client):
    response = client.post("/v1/completions", json={"max_tokens": 10}) # Missing prompt
    assert response.status_code == 422 # Unprocessable Entity
