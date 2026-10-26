import pytest
from server.app import speculative_engine, eviction_engine, batching_engine, hf_engine


def test_engines_initialization():
    assert hf_engine.model is not None
    assert hf_engine.tokenizer is not None
    assert speculative_engine.model is hf_engine.model
    assert eviction_engine.model is hf_engine.model
    assert batching_engine.model is hf_engine.model


@pytest.mark.asyncio
async def test_speculative_engine_telemetry():
    telemetry = speculative_engine.get_telemetry()
    assert "acceptance_rate" in telemetry
    assert "tokens_per_step" in telemetry
    assert telemetry["k"] == 4


@pytest.mark.asyncio
async def test_eviction_engine_telemetry():
    telemetry = eviction_engine.get_telemetry()
    assert "budget_tokens" in telemetry
    assert "num_sink_tokens" in telemetry
    assert "total_tokens_evicted" in telemetry
