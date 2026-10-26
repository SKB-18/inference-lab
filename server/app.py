import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Engines

from engine.hf_engine import HFEngine
from engine.batching_queue import BatchingEngine
from engine.speculative import SpeculativeEngine
from engine.cache_eviction import KVCacheEvictionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize production engines sharing single PyTorch model instance
hf_engine = HFEngine()
batching_engine = BatchingEngine(max_batch_size=16, model=hf_engine.model, tokenizer=hf_engine.tokenizer)
speculative_engine = SpeculativeEngine(model=hf_engine.model, tokenizer=hf_engine.tokenizer, k=4)
eviction_engine = KVCacheEvictionEngine(model=hf_engine.model, tokenizer=hf_engine.tokenizer, budget_tokens=48)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting background batching scheduler...")
    await batching_engine.start()
    yield
    logger.info("Shutting down background batching scheduler...")
    await batching_engine.stop()

app = FastAPI(
    title="Inference Lab - Production Local LLM Engine",
    description="High-performance serving engine with real HuggingFace models, continuous batching, and KV-cache optimization.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=100, ge=0, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    mode: str = "real"  # 'real' (Qwen2.5-0.5B), 'batching', 'speculative', 'eviction'

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=150, ge=0, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engine": "HFEngine",
        "model": hf_engine.model_name,
        "device": hf_engine.device
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": hf_engine.model_name,
                "object": "model",
                "owned_by": "local-inference-lab"
            }
        ]
    }

@app.get("/v1/telemetry")
def get_telemetry():
    return {
        "model": hf_engine.model_name,
        "device": hf_engine.device,
        "batching": {
            "total_tokens": batching_engine.total_tokens_generated,
            "total_batches": batching_engine.total_batches_processed
        },
        "speculative": speculative_engine.get_telemetry(),
        "cache_eviction": eviction_engine.get_telemetry()
    }


@app.post("/v1/completions")
async def completions(request: GenerateRequest):
    start_time = time.time()
    logger.info(f"Incoming completion request: mode={request.mode}, max_tokens={request.max_tokens}")
    
    # Select engine
    if request.mode == "speculative":
        stream_source = speculative_engine.generate_stream(request.prompt, request.max_tokens)
    elif request.mode == "eviction":
        stream_source = eviction_engine.generate_stream(request.prompt, request.max_tokens)
    elif request.mode == "batching":
        stream_source = batching_engine.generate_stream(request.prompt, request.max_tokens)
    else:
        # Default to real production HuggingFace model
        stream_source = hf_engine.generate_stream(
            request.prompt, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
    
    async def stream_generator():
        first_token = True
        token_count = 0
        
        async for token in stream_source:
            if first_token:
                ttft = time.time() - start_time
                logger.info(f"Time to First Token (TTFT): {ttft:.4f}s")
                first_token = False
            
            token_count += 1
            yield token
            
        total_time = time.time() - start_time
        tps = token_count / total_time if total_time > 0 else 0
        logger.info(f"Request completed: tokens={token_count}, total_time={total_time:.4f}s, tps={tps:.2f} tok/s")

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible /v1/chat/completions endpoint for agent frameworks and standard clients.
    """
    messages_payload = [{"role": m.role, "content": m.content} for m in request.messages]
    
    if hasattr(hf_engine.tokenizer, "apply_chat_template") and hf_engine.tokenizer.chat_template:
        formatted_prompt = hf_engine.tokenizer.apply_chat_template(
            messages_payload, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages) + "\nassistant:"

    generated_text = ""
    async for token in hf_engine.generate_stream(
        prompt=formatted_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    ):
        generated_text += token

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": hf_engine.model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text.strip()
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(hf_engine.tokenizer.encode(formatted_prompt)),
            "completion_tokens": len(hf_engine.tokenizer.encode(generated_text)),
            "total_tokens": len(hf_engine.tokenizer.encode(formatted_prompt)) + len(hf_engine.tokenizer.encode(generated_text))
        }
    }

