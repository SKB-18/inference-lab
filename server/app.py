import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# In M2, we use BatchingEngine for continuous batching simulation
from engine.batching_queue import BatchingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

engine = BatchingEngine(max_batch_size=16)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the continuous batching worker loop
    await engine.start()
    yield
    # Cleanup on shutdown
    await engine.stop()

app = FastAPI(title="Inference Lab - M2 Continuous Batching", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50

@app.post("/v1/completions")
async def completions(request: GenerateRequest):
    start_time = time.time()
    
    async def stream_generator():
        first_token = True
        token_count = 0
        
        async for token in engine.generate_stream(request.prompt, request.max_tokens):
            if first_token:
                ttft = time.time() - start_time
                logger.info(f"Time to First Token (TTFT): {ttft:.4f}s")
                first_token = False
            
            token_count += 1
            yield token
            
        total_time = time.time() - start_time
        tps = token_count / total_time
        logger.info(f"Request finished. Tokens: {token_count}, Total Time: {total_time:.4f}s, TPS: {tps:.2f} tok/s")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
