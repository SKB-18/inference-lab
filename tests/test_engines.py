import pytest
from engine.speculative import SpeculativeEngine
from engine.cache_eviction import KVCacheEvictionEngine

@pytest.mark.asyncio
async def test_speculative_engine_generation():
    engine = SpeculativeEngine(target_latency=0.01, draft_latency=0.001, k=4)
    tokens = []
    async for token in engine.generate_stream("Test prompt", max_tokens=10):
        tokens.append(token)
        
    assert len(tokens) == 10
    assert "Test" in tokens[0]

@pytest.mark.asyncio
async def test_eviction_engine_generation():
    # Set a tiny budget to force eviction immediately
    engine = KVCacheEvictionEngine(budget_mb=1)
    tokens = []
    async for token in engine.generate_stream("Test long context prompt " * 10, max_tokens=15):
        tokens.append(token)
        
    assert len(tokens) == 15
    # Ensure VRAM metric is embedded in output
    assert "[VRAM:" in tokens[-1]
