import asyncio
import time
import logging
import random
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class SpeculativeEngine:
    """
    Simulates Speculative Decoding (Draft-Target verification).
    In speculative decoding, a smaller 'draft' model autoregressively generates K tokens.
    A larger 'target' model then evaluates these K tokens in a single forward pass.
    If the target model agrees with the draft, we get K tokens for the latency of 1 step.
    This massively increases Tokens Per Second (TPS).
    """
    def __init__(self, target_latency: float = 0.05, draft_latency: float = 0.01, k: int = 4):
        self.target_latency = target_latency
        self.draft_latency = draft_latency
        self.k = k # Number of draft tokens to guess
        logger.info(f"SpeculativeEngine initialized. K={k}, Draft Latency={draft_latency}s, Target Latency={target_latency}s")

    async def generate_stream(self, prompt: str, max_tokens: int = 50) -> AsyncGenerator[str, None]:
        # Simulate TTFT (Time To First Token) for the target model
        await asyncio.sleep(0.2)
        
        words = prompt.split() + ["speculative", "decoding", "speedup", "test"] * (max_tokens // 4 + 1)
        tokens_generated = 0
        
        while tokens_generated < max_tokens:
            # 1. Draft model generates K tokens autoregressively
            draft_time = self.draft_latency * self.k
            await asyncio.sleep(draft_time)
            
            # 2. Target model does a single forward pass to verify
            await asyncio.sleep(self.target_latency)
            
            # 3. Acceptance rate (simulated)
            # Typically 60-80% of draft tokens are accepted depending on alignment
            acceptance_rate = random.uniform(0.5, 0.9)
            accepted_tokens = max(1, int(self.k * acceptance_rate))
            
            # Yield the accepted tokens rapidly to the client
            for _ in range(accepted_tokens):
                if tokens_generated < max_tokens:
                    word = words[tokens_generated]
                    yield f"{word} "
                    tokens_generated += 1
                
                # Tiny simulated network delay between yielding accepted tokens
                await asyncio.sleep(0.001)
