import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

@dataclass
class BatchedRequest:
    request_id: str
    prompt: str
    input_ids: List[int]
    max_tokens: int
    tokens_generated: int = 0
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None

class BatchingEngine:
    """
    Production Continuous Batching Engine (Iteration-Level Scheduling).
    
    Maintains a dynamic queue of incoming concurrent requests. On each decode step,
    active requests are batched together into a single PyTorch forward pass with left-padding.
    As requests complete (EOS or max_tokens), they are retired immediately, and new requests
    from the queue swap into the active batch without waiting for the rest of the batch to finish.
    """
    
    def __init__(
        self, 
        max_batch_size: int = 16, 
        model=None, 
        tokenizer=None, 
        device: Optional[str] = None
    ):
        self.max_batch_size = max_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            logger.info(f"Loading standalone model {model_name} for BatchingEngine...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            ).to(self.device)
            self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.queue: asyncio.Queue[BatchedRequest] = asyncio.Queue()
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.request_counter = 0
        
        # Telemetry metrics
        self.total_tokens_generated = 0
        self.total_batches_processed = 0

    async def start(self):
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._worker_loop())
            logger.info(f"Continuous Batching Engine started with max_batch_size={self.max_batch_size}")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()

    async def generate_stream(self, prompt: str, max_tokens: int = 50) -> AsyncGenerator[str, None]:
        if max_tokens <= 0:
            return

        self.request_counter += 1
        req_id = f"req-{self.request_counter}"
        
        # Format with chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt
            
        input_ids = self.tokenizer.encode(formatted, add_special_tokens=True)
        req = BatchedRequest(
            request_id=req_id,
            prompt=prompt,
            input_ids=input_ids,
            max_tokens=max_tokens
        )
        
        await self.queue.put(req)
        
        while True:
            token = await req.queue.get()
            if token is None:  # EOS / termination sentinel
                break
            yield token

    async def _worker_loop(self):
        active_batch: List[BatchedRequest] = []
        loop = asyncio.get_running_loop()
        
        while self.running:
            # 1. Fill active batch up to max_batch_size with waiting requests
            while len(active_batch) < self.max_batch_size:
                try:
                    req = self.queue.get_nowait()
                    active_batch.append(req)
                except asyncio.QueueEmpty:
                    break
            
            if not active_batch:
                await asyncio.sleep(0.005)
                continue
                
            # 2. Run batched forward step in executor thread so async event loop stays unblocked
            next_batch = await loop.run_in_executor(None, self._step_batch, active_batch)
            active_batch = next_batch
            self.total_batches_processed += 1

    def _step_batch(self, active_batch: List[BatchedRequest]) -> List[BatchedRequest]:
        if not active_batch:
            return []

        # Prepare left-padded tensors
        max_len = max(len(req.input_ids) for req in active_batch)
        pad_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []

        for req in active_batch:
            pad_len = max_len - len(req.input_ids)
            padded_ids = [pad_id] * pad_len + req.input_ids
            mask = [0] * pad_len + [1] * len(req.input_ids)
            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(mask)

        input_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(batch_attention_mask, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, attention_mask=mask_tensor)
            # Sample next token from the last position across each row
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1).tolist()

        next_batch = []
        eos_id = self.tokenizer.eos_token_id

        for req, next_tok in zip(active_batch, next_tokens):
            now = time.time()
            if req.first_token_time is None:
                req.first_token_time = now

            token_str = self.tokenizer.decode([next_tok])
            req.queue.put_nowait(token_str)
            req.input_ids.append(next_tok)
            req.tokens_generated += 1
            self.total_tokens_generated += 1

            # Check retirement condition (EOS or max_tokens reached)
            if next_tok == eos_id or req.tokens_generated >= req.max_tokens:
                req.queue.put_nowait(None)  # Signal completion to client stream
            else:
                next_batch.append(req)

        return next_batch
