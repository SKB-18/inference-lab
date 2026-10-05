from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_completions_streaming():
    response = client.post(
        "/v1/completions", 
        json={"prompt": "Hello world", "max_tokens": 5}
    )
    assert response.status_code == 200
    # FastAPI streaming response type
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # Collect streamed bytes
    content = b""
    for chunk in response.iter_bytes():
        content += chunk
        
    assert len(content) > 0
    decoded = content.decode("utf-8")
    assert "Hello" in decoded or "simulated" in decoded
