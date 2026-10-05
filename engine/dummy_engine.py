import asyncio
import time
from typing import AsyncGenerator
from .base import BaseEngine

class DummyEngine(BaseEngine):
    """
    A simulated engine that yields tokens with artificial latency.
    Used for testing the server architecture, TTFT, and batching queues 
    without the overhead of loading actual ML weights.
    """
    
    async def generate_stream(self, prompt: str, max_tokens: int = 100) -> AsyncGenerator[str, None]:
        # Simulate processing time before the first token (TTFT)
        await asyncio.sleep(0.2)
        
        words = prompt.split() + ["simulated", "token", "stream", "test"] * (max_tokens // 4)
        for i, word in enumerate(words[:max_tokens]):
            # Simulate per-token generation latency (inter-token latency)
            await asyncio.sleep(0.05)
            yield f"{word} "
