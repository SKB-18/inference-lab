import asyncio
import time
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class BatchingEngine:
    """
    Simulates continuous batching.
    Requests are placed in a queue. A background worker loop pulls requests,
    up to `max_batch_size`. It performs a "forward pass" step (simulated by sleep),
    then yields a token for every active request in the batch. As requests finish,
    they are removed and new ones are immediately added from the queue without waiting
    for the entire batch to complete.
    """
    def __init__(self, max_batch_size: int = 16):
        self.max_batch_size = max_batch_size
        self.queue = asyncio.Queue()
        self.running = False
        self.task = None

    async def start(self):
        self.running = True
        self.task = asyncio.create_task(self._worker_loop())
        logger.info(f"Batching Engine started with max_batch_size={self.max_batch_size}")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()

    async def generate_stream(self, prompt: str, max_tokens: int = 50) -> AsyncGenerator[str, None]:
        # Communication channel for the worker to send back generated tokens
        response_queue = asyncio.Queue()
        
        request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "tokens_generated": 0,
            "queue": response_queue,
            # Simple text generation simulation
            "words": prompt.split() + ["batch", "token", "stream", "test"] * (max_tokens // 4 + 1)
        }
        
        await self.queue.put(request)
        
        while True:
            token = await response_queue.get()
            if token is None: # EOS
                break
            yield token

    async def _worker_loop(self):
        active_batch = []
        
        while self.running:
            # 1. Fill the batch up to max_batch_size using available requests in queue
            while len(active_batch) < self.max_batch_size:
                try:
                    req = self.queue.get_nowait()
                    active_batch.append(req)
                except asyncio.QueueEmpty:
                    break
            
            if not active_batch:
                # Idle: wait briefly for incoming requests
                await asyncio.sleep(0.01)
                continue
                
            # 2. Simulate forward pass (batch computation)
            # A larger batch might take slightly longer, but much less than sequential
            await asyncio.sleep(0.05 + 0.002 * len(active_batch))
            
            # 3. Yield next token for each request
            next_batch = []
            for req in active_batch:
                word = req["words"][req["tokens_generated"]]
                # put_nowait so we don't block the worker loop
                req["queue"].put_nowait(f"{word} ")
                req["tokens_generated"] += 1
                
                # Check for EOS
                if req["tokens_generated"] >= req["max_tokens"]:
                    req["queue"].put_nowait(None)
                else:
                    next_batch.append(req)
                    
            active_batch = next_batch
