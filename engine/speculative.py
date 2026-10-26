import asyncio
import time
import logging
from typing import AsyncGenerator, List, Optional, Tuple, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.base import BaseEngine

logger = logging.getLogger(__name__)


class PromptLookupProposer:
    """
    Prompt Lookup Decoding (PLD / N-Gram Candidate Matching).
    
    Inspects the prompt and previously generated tokens to find recurring n-grams.
    If a suffix match of length n is discovered earlier in the sequence, proposes
    the subsequent K tokens as candidates for speculative verification.
    Extremely fast: zero model overhead, pure array/hash lookup.
    """
    def __init__(self, max_ngram_size: int = 3, min_ngram_size: int = 1):
        self.max_ngram_size = max_ngram_size
        self.min_ngram_size = min_ngram_size

    def propose(self, sequence: List[int], k: int = 4) -> List[int]:
        if len(sequence) < self.min_ngram_size + 1:
            return []

        seq_len = len(sequence)
        
        # Search from largest ngram down to smallest
        for ngram_size in range(min(self.max_ngram_size, seq_len - 1), self.min_ngram_size - 1, -1):
            target_ngram = sequence[-ngram_size:]
            
            # Look backwards from the start of the sequence up to seq_len - ngram_size
            for i in range(seq_len - ngram_size - 1):
                if sequence[i : i + ngram_size] == target_ngram:
                    # Found matching n-gram! Extract the subsequent K tokens
                    start_idx = i + ngram_size
                    end_idx = min(start_idx + k, seq_len)
                    candidates = sequence[start_idx:end_idx]
                    if candidates:
                        return candidates

        return []


class SpeculativeEngine(BaseEngine):
    """
    Production Speculative Decoding Engine.
    
    Uses a fast proposer (Prompt Lookup / N-gram candidate generation or draft model)
    to propose K candidate tokens, followed by a parallel single forward pass verification
    on the target model.
    
    Accepts matching tokens up to the first divergence, emits the target model's correction
    token, and samples a bonus token when all K draft tokens match.
    """
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        k: int = 4,
        device: Optional[str] = None,
        proposer: Optional[PromptLookupProposer] = None
    ):
        self.k = k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proposer = proposer or PromptLookupProposer(max_ngram_size=3, min_ngram_size=1)
        
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            logger.info(f"[SpeculativeEngine] Loading standalone model {model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            ).to(self.device)
            self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Telemetry & performance metrics
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.total_target_forward_passes = 0
        self.total_tokens_generated = 0
        self.total_speculative_runs = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.total_draft_tokens

    @property
    def tokens_per_step(self) -> float:
        if self.total_target_forward_passes == 0:
            return 0.0
        return self.total_tokens_generated / self.total_target_forward_passes

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "total_draft_tokens": self.total_draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_forward_passes": self.total_target_forward_passes,
            "total_tokens_generated": self.total_tokens_generated,
            "tokens_per_step": round(self.tokens_per_step, 2),
            "k": self.k
        }

    def verify_draft(
        self, 
        prefix_ids: List[int], 
        draft_ids: List[int]
    ) -> Tuple[List[int], int, bool]:
        """
        Executes a single parallel verification forward pass on the target model.
        
        Returns:
            (emitted_tokens, num_accepted_draft, all_accepted)
        """
        combined = prefix_ids + draft_ids
        input_tensor = torch.tensor([combined], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor)
            logits = outputs.logits[0]  # Shape: (seq_len, vocab_size)

        self.total_target_forward_passes += 1
        
        N = len(prefix_ids)
        K = len(draft_ids)
        
        emitted_tokens = []
        num_accepted_draft = 0
        all_accepted = True
        
        for i in range(K):
            # Target prediction at position N - 1 + i corresponds to token at N + i
            target_pred = torch.argmax(logits[N - 1 + i, :]).item()
            draft_tok = draft_ids[i]
            
            if target_pred == draft_tok:
                # Draft token verified and accepted!
                emitted_tokens.append(draft_tok)
                num_accepted_draft += 1
            else:
                # Mismatch! Accept target model's correction token, discard remaining draft
                emitted_tokens.append(target_pred)
                all_accepted = False
                break
                
        # If all K draft tokens were accepted, target model produces a bonus token at position N + K - 1
        if all_accepted:
            bonus_token = torch.argmax(logits[N + K - 1, :]).item()
            emitted_tokens.append(bonus_token)

        self.total_draft_tokens += K
        self.accepted_draft_tokens += num_accepted_draft
        
        return emitted_tokens, num_accepted_draft, all_accepted

    def single_step(self, sequence_ids: List[int]) -> int:
        """Standard 1-token autoregressive target forward step fallback."""
        input_tensor = torch.tensor([sequence_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor)
            next_token = torch.argmax(outputs.logits[0, -1, :]).item()
            
        self.total_target_forward_passes += 1
        return next_token

    async def generate_stream(
        self, 
        prompt: str, 
        max_tokens: int = 100
    ) -> AsyncGenerator[str, None]:
        """
        Streams generated tokens using real speculative verification.
        Preserves non-blocking async execution using background thread executor.
        """
        if max_tokens <= 0:
            return

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        current_sequence = self.tokenizer.encode(formatted, add_special_tokens=True)
        tokens_generated = 0
        eos_id = self.tokenizer.eos_token_id
        loop = asyncio.get_running_loop()
        self.total_speculative_runs += 1

        while tokens_generated < max_tokens:
            # 1. Propose draft tokens
            draft_candidates = self.proposer.propose(current_sequence, k=self.k)
            
            # Limit candidate length so we don't overshoot max_tokens
            remaining = max_tokens - tokens_generated
            if len(draft_candidates) > remaining:
                draft_candidates = draft_candidates[:remaining]

            if draft_candidates:
                # 2. Speculative verification pass
                emitted_tokens, accepted_count, _ = await loop.run_in_executor(
                    None,
                    self.verify_draft,
                    current_sequence,
                    draft_candidates
                )
            else:
                # 3. Fallback: single autoregressive step
                next_tok = await loop.run_in_executor(
                    None,
                    self.single_step,
                    current_sequence
                )
                emitted_tokens = [next_tok]

            # Yield each emitted token rapidly to the client
            hit_eos = False
            for tok_id in emitted_tokens:
                if tokens_generated >= max_tokens:
                    break
                    
                current_sequence.append(tok_id)
                tokens_generated += 1
                self.total_tokens_generated += 1
                
                text_chunk = self.tokenizer.decode([tok_id])
                yield text_chunk
                
                if tok_id == eos_id:
                    hit_eos = True
                    break

            if hit_eos:
                break
