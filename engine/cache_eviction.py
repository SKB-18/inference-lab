import asyncio
import time
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class KVCacheEvictionEngine:
    """
    Simulates KV-Cache Eviction (Heavy-Hitter / H2O style).
    For long contexts, the KV cache grows linearly, causing OOM (Out of Memory).
    This engine simulates an eviction policy capped at a specific memory budget.
    It tracks 'simulated VRAM' usage.
    """
    def __init__(self, budget_mb: int = 1024):
        self.budget_mb = budget_mb
        logger.info(f"KVCacheEvictionEngine initialized with budget: {budget_mb} MB")

    async def generate_stream(self, prompt: str, max_tokens: int = 50) -> AsyncGenerator[str, None]:
        await asyncio.sleep(0.2) # TTFT
        
        words = prompt.split() + ["kv", "cache", "eviction", "saved", "vram"] * (max_tokens // 4 + 1)
        tokens_generated = 0
        
        # Simulate base VRAM (model weights) + Context length
        base_vram = 4000
        prompt_vram_cost = len(prompt.split()) * 0.5 # 0.5 MB per token simulated
        current_vram = base_vram + prompt_vram_cost
        
        while tokens_generated < max_tokens:
            await asyncio.sleep(0.02) # Fast generation step
            
            # Simulate growing KV cache
            current_vram += 0.5 
            
            # Eviction logic: if we exceed budget (on top of base_vram), evict!
            if (current_vram - base_vram) > self.budget_mb:
                evicted_amount = (current_vram - base_vram) - self.budget_mb
                logger.debug(f"Evicting {evicted_amount} MB of KV cache to stay within budget.")
                current_vram -= evicted_amount # Force it back to budget
                
            word = words[tokens_generated]
            # We yield a special token indicating VRAM usage for the UI to parse
            yield f"{word} [VRAM:{current_vram:.1f}MB] "
            tokens_generated += 1
