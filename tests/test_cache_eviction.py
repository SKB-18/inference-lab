import pytest
import torch
from transformers.cache_utils import DynamicCache
from engine.cache_eviction import KVCacheEvictionEngine


def test_eviction_engine_init():
    engine = KVCacheEvictionEngine(budget_tokens=32, num_sink_tokens=4)
    assert engine.budget_tokens == 32
    assert engine.num_sink_tokens == 4
    assert engine.window_size == 28


def test_eviction_under_budget():
    engine = KVCacheEvictionEngine(budget_tokens=20, num_sink_tokens=4)
    cache = DynamicCache()
    # Mock layer with 10 tokens (less than budget 20)
    cache.update(torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8), 0)
    
    evicted = engine.evict_cache(cache)
    assert evicted == 0
    assert cache.layers[0].keys.shape[2] == 10


def test_eviction_over_budget():
    engine = KVCacheEvictionEngine(budget_tokens=10, num_sink_tokens=2)
    cache = DynamicCache()
    # Mock layer with 20 tokens (over budget 10)
    # Budget=10, Sink=2, Window=8 -> Should keep [0, 1] + [12, 13, ..., 19] = 10 tokens
    cache.update(torch.randn(1, 2, 20, 8), torch.randn(1, 2, 20, 8), 0)
    
    evicted = engine.evict_cache(cache)
    assert evicted == 10
    assert cache.layers[0].keys.shape[2] == 10
    assert cache.layers[0].values.shape[2] == 10


def test_memory_computation():
    engine = KVCacheEvictionEngine(budget_tokens=20, num_sink_tokens=4)
    cache = DynamicCache()
    # 1 batch * 2 heads * 10 seq_len * 8 dim * 4 bytes/float32 = 640 bytes for keys + 640 bytes for values = 1280 bytes
    cache.update(torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8), 0)
    mem_mb = engine.compute_cache_memory_mb(cache)
    assert mem_mb > 0.0
    assert mem_mb < 1.0


@pytest.mark.asyncio
async def test_eviction_engine_real_generation():
    from server.app import eviction_engine
    
    # Prompt with some length
    prompt = "Explain in high detail the concept of Key-Value Cache Eviction and attention sinks in Large Language Models."
    tokens = []
    async for tok in eviction_engine.generate_stream(prompt, max_tokens=15):
        tokens.append(tok)
        
    assert len(tokens) > 0
    telemetry = eviction_engine.get_telemetry()
    assert telemetry["total_tokens_generated"] > 0
    assert "peak_vram_mb" in telemetry
