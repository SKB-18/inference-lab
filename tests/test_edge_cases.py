import pytest
import asyncio
from fastapi.testclient import TestClient
from server.app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_empty_prompt(client):
    response = client.post("/v1/completions", json={"prompt": "", "max_tokens": 10})
    assert response.status_code == 200
    # Even if empty, the simulation should return some generated tokens based on dummy engine
    assert len(response.content) > 0

def test_zero_max_tokens(client):
    response = client.post("/v1/completions", json={"prompt": "Hello", "max_tokens": 0})
    assert response.status_code == 200
    # Dummy engine might return the prompt word
    assert b"Hello" in response.content

def test_huge_max_tokens(client):
    # Testing an edge case where max_tokens is unusually large, but limit for test speed
    response = client.post("/v1/completions", json={"prompt": "Test", "max_tokens": 100})
    assert response.status_code == 200
    assert len(response.content) > 50

def test_missing_fields(client):
    response = client.post("/v1/completions", json={"max_tokens": 10}) # Missing prompt
    assert response.status_code == 422 # Unprocessable Entity (FastAPI validation)
