import asyncio
import time
import logging
from typing import AsyncGenerator, Optional, Dict, Any, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from engine.base import BaseEngine

logger = logging.getLogger(__name__)


class KVCacheEvictionEngine(BaseEngine):
    """
    Production-grade KV-Cache Eviction Engine based on StreamingLLM & Attention Sinks.
    
    In autoregressive decoding, the Key-Value (KV) cache grows linearly O(N), leading
    to VRAM/RAM exhaustion on long contexts or high concurrency.
    
    This engine enforces a hard budget on the KV-cache sequence length:
    1. Attention Sink Tokens (first 4 tokens) are permanently retained to prevent perplexity collapse.
    2. A Sliding Window of the most recent W tokens is preserved for local syntax and coherence.
    3. Intermediate tokens are evicted directly from the PyTorch key/value tensors along dim=2.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        budget_tokens: int = 48,
        num_sink_tokens: int = 4,
        device: Optional[str] = None
    ):
        self.budget_tokens = max(budget_tokens, num_sink_tokens + 2)
        self.num_sink_tokens = num_sink_tokens
        self.window_size = self.budget_tokens - self.num_sink_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            logger.info(f"[KVCacheEvictionEngine] Loading standalone model {model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            ).to(self.device)
            self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Telemetry metrics
        self.total_tokens_generated = 0
        self.total_eviction_events = 0
        self.total_tokens_evicted = 0
        self.peak_vram_mb = 0.0

    def compute_cache_memory_mb(self, past_key_values: DynamicCache) -> float:
        """Calculates exact memory footprint of the KV-cache tensors in megabytes."""
        if not hasattr(past_key_values, "layers"):
            return 0.0

        total_bytes = 0
        for layer in past_key_values.layers:
            if getattr(layer, "keys", None) is not None:
                total_bytes += layer.keys.nelement() * layer.keys.element_size()
            if getattr(layer, "values", None) is not None:
                total_bytes += layer.values.nelement() * layer.values.element_size()

        return total_bytes / (1024 * 1024)

    def evict_cache(self, past_key_values: DynamicCache) -> int:
        """
        Prunes key and value tensors across all transformer layers to stay within budget.
        Returns the number of tokens evicted in this event.
        """
        if not hasattr(past_key_values, "layers") or not past_key_values.layers:
            return 0

        first_layer = past_key_values.layers[0]
        if getattr(first_layer, "keys", None) is None:
            return 0

        curr_seq_len = first_layer.keys.shape[2]
        if curr_seq_len <= self.budget_tokens:
            return 0

        # Construct keep index mask: sink tokens [0..sink-1] + sliding window [curr_len - window..curr_len-1]
        sink_indices = list(range(0, self.num_sink_tokens))
        recent_indices = list(range(curr_seq_len - self.window_size, curr_seq_len))
        keep_indices_list = sorted(list(set(sink_indices + recent_indices)))
        
        tokens_removed = curr_seq_len - len(keep_indices_list)

        device = first_layer.keys.device
        idx_tensor = torch.tensor(keep_indices_list, dtype=torch.long, device=device)

        for layer in past_key_values.layers:
            if getattr(layer, "keys", None) is not None and getattr(layer, "values", None) is not None:
                layer.keys = layer.keys[:, :, idx_tensor, :]
                layer.values = layer.values[:, :, idx_tensor, :]

        self.total_eviction_events += 1
        self.total_tokens_evicted += tokens_removed
        return tokens_removed

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "budget_tokens": self.budget_tokens,
            "num_sink_tokens": self.num_sink_tokens,
            "window_size": self.window_size,
            "total_tokens_generated": self.total_tokens_generated,
            "total_eviction_events": self.total_eviction_events,
            "total_tokens_evicted": self.total_tokens_evicted,
            "peak_vram_mb": round(self.peak_vram_mb, 3)
        }

    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 100
    ) -> AsyncGenerator[str, None]:
        """
        Autoregressively streams generated tokens with active KV-cache attention sink eviction.
        """
        if max_tokens <= 0:
            return

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        loop = asyncio.get_running_loop()
        eos_id = self.tokenizer.eos_token_id

        # 1. Prefill stage: single forward pass over prompt to build initial KV-cache
        def prefill_step():
            with torch.no_grad():
                out = self.model(input_ids=input_ids, use_cache=True)
                next_tok = torch.argmax(out.logits[0, -1, :]).item()
                pkv = out.past_key_values
                return next_tok, pkv

        next_tok, past_key_values = await loop.run_in_executor(None, prefill_step)

        # Check if prompt exceeded budget already
        await loop.run_in_executor(None, self.evict_cache, past_key_values)

        token_str = self.tokenizer.decode([next_tok])
        self.total_tokens_generated += 1
        yield token_str

        tokens_generated = 1

        # 2. Decode stage: step-by-step with KV-cache eviction
        while tokens_generated < max_tokens and next_tok != eos_id:
            # Check eviction condition before or after step
            evicted = await loop.run_in_executor(None, self.evict_cache, past_key_values)
            
            def decode_step(curr_tok, pkv):
                step_input = torch.tensor([[curr_tok]], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    out = self.model(input_ids=step_input, past_key_values=pkv, use_cache=True)
                    nxt = torch.argmax(out.logits[0, -1, :]).item()
                    mem_mb = self.compute_cache_memory_mb(out.past_key_values)
                    return nxt, out.past_key_values, mem_mb

            next_tok, past_key_values, mem_mb = await loop.run_in_executor(
                None, decode_step, next_tok, past_key_values
            )

            if mem_mb > self.peak_vram_mb:
                self.peak_vram_mb = mem_mb

            token_str = self.tokenizer.decode([next_tok])
            tokens_generated += 1
            self.total_tokens_generated += 1

            yield token_str

            if next_tok == eos_id:
                break
