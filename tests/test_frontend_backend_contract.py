import os
import requests
import pytest

BASE_URL = "http://127.0.0.1:8002"
FRONTEND_URL = "http://localhost:5173"


def test_frontend_dev_server_serves_html():
    """Validates that the Vite frontend dev server is active on port 5173 and serving index.html."""
    res = requests.get(FRONTEND_URL, timeout=5)
    assert res.status_code == 200
    assert '<div id="root"></div>' in res.text
    assert "src/main.jsx" in res.text or "@vite/client" in res.text


def test_frontend_telemetry_schema():
    """Validates the exact telemetry schema consumed by App.jsx in the frontend."""
    res = requests.get(f"{BASE_URL}/v1/telemetry", timeout=5)
    assert res.status_code == 200
    data = res.json()
    
    # Root keys
    assert "model" in data
    assert "device" in data
    assert "speculative" in data
    assert "cache_eviction" in data
    assert "batching" in data
    
    # Speculative fields accessed by UI (lines 190-192 of App.jsx)
    spec = data["speculative"]
    assert "acceptance_rate" in spec
    assert "tokens_per_step" in spec
    assert "total_draft_tokens" in spec
    assert isinstance(spec["acceptance_rate"], (int, float))
    assert isinstance(spec["tokens_per_step"], (int, float))
    assert isinstance(spec["total_draft_tokens"], int)
    
    # Cache eviction fields accessed by UI (lines 197-200 of App.jsx)
    evict = data["cache_eviction"]
    assert "budget_tokens" in evict
    assert "num_sink_tokens" in evict
    assert "total_tokens_evicted" in evict
    assert "peak_vram_mb" in evict
    assert isinstance(evict["budget_tokens"], int)
    assert isinstance(evict["num_sink_tokens"], int)
    assert isinstance(evict["total_tokens_evicted"], int)
    assert isinstance(evict["peak_vram_mb"], (int, float))
    
    # Batching fields accessed by UI (lines 205-206 of App.jsx)
    batch = data["batching"]
    assert "total_tokens" in batch
    assert "total_batches" in batch
    assert isinstance(batch["total_tokens"], int)
    assert isinstance(batch["total_batches"], int)


def test_frontend_completions_call_real():
    """Tests /v1/completions with the exact payload structure sent by the React frontend."""
    payload = {
        "prompt": "Hello",
        "max_tokens": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "mode": "real"
    }
    response = requests.post(f"{BASE_URL}/v1/completions", json=payload, stream=True, timeout=15)
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    
    stream_content = response.text
    assert len(stream_content) > 0


def test_frontend_completions_call_batching():
    """Tests frontend batching mode endpoint invocation."""
    payload = {
        "prompt": "Test batching",
        "max_tokens": 4,
        "temperature": 0.5,
        "top_p": 0.9,
        "mode": "batching"
    }
    response = requests.post(f"{BASE_URL}/v1/completions", json=payload, stream=True, timeout=15)
    assert response.status_code == 200
    assert len(response.text) > 0


def test_frontend_completions_call_speculative():
    """Tests frontend speculative decoding mode endpoint invocation."""
    payload = {
        "prompt": "The quick brown fox",
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "mode": "speculative"
    }
    response = requests.post(f"{BASE_URL}/v1/completions", json=payload, stream=True, timeout=15)
    assert response.status_code == 200
    assert len(response.text) > 0


def test_frontend_completions_call_eviction():
    """Tests frontend KV-cache eviction mode endpoint invocation."""
    payload = {
        "prompt": "Attention sinks test",
        "max_tokens": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "mode": "eviction"
    }
    response = requests.post(f"{BASE_URL}/v1/completions", json=payload, stream=True, timeout=15)
    assert response.status_code == 200
    assert len(response.text) > 0


def test_frontend_slider_boundary_values():
    """Tests frontend slider boundary extremes (min/max sliders)."""
    min_payload = {
        "prompt": "Boundary test",
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 0.05,
        "mode": "real"
    }
    res_min = requests.post(f"{BASE_URL}/v1/completions", json=min_payload, timeout=10)
    assert res_min.status_code == 200

    max_payload = {
        "prompt": "Boundary test",
        "max_tokens": 2,
        "temperature": 2.0,
        "top_p": 1.0,
        "mode": "real"
    }
    res_max = requests.post(f"{BASE_URL}/v1/completions", json=max_payload, timeout=10)
    assert res_max.status_code == 200


def test_frontend_dist_build_artifacts():
    """Verifies that the compiled Vite production assets exist and are non-empty."""
    dist_dir = os.path.join(os.path.dirname(__file__), "..", "ui", "dist")
    index_html = os.path.join(dist_dir, "index.html")
    assert os.path.exists(index_html), "dist/index.html was not generated!"
    with open(index_html, "r", encoding="utf-8") as f:
        html = f.read()
    assert '<div id="root"></div>' in html
    assert "assets/" in html
