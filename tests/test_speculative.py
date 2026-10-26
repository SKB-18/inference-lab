import pytest
import torch
from engine.speculative import PromptLookupProposer, SpeculativeEngine


def test_prompt_lookup_proposer_empty_and_short():
    proposer = PromptLookupProposer(max_ngram_size=3, min_ngram_size=1)
    # Sequence too short
    assert proposer.propose([], k=4) == []
    assert proposer.propose([1], k=4) == []


def test_prompt_lookup_proposer_pattern_matching():
    proposer = PromptLookupProposer(max_ngram_size=3, min_ngram_size=2)
    # Sequence: [10, 20, 30, 40, 50, 10, 20]
    # Suffix is [10, 20]. Earlier match has candidates [30, 40, 50]
    seq = [10, 20, 30, 40, 50, 10, 20]
    candidates = proposer.propose(seq, k=3)
    assert candidates == [30, 40, 50]


def test_prompt_lookup_proposer_k_truncation():
    proposer = PromptLookupProposer(max_ngram_size=2, min_ngram_size=1)
    seq = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
    candidates = proposer.propose(seq, k=2)
    assert len(candidates) == 2
    assert candidates == [3, 4]


def test_prompt_lookup_proposer_no_match():
    proposer = PromptLookupProposer(max_ngram_size=2, min_ngram_size=2)
    seq = [1, 2, 3, 4, 5, 6]
    candidates = proposer.propose(seq, k=4)
    assert candidates == []


@pytest.mark.asyncio
async def test_speculative_engine_verification_math():
    # Load shared engine
    from server.app import speculative_engine, hf_engine
    
    # Real prompt: "The quick brown fox"
    prefix_ids = hf_engine.tokenizer.encode("The quick brown fox", add_special_tokens=True)
    
    # 1. Propose draft tokens that diverge
    draft_ids = [999999 % hf_engine.tokenizer.vocab_size, 100]
    emitted, accepted_count, all_acc = speculative_engine.verify_draft(prefix_ids, draft_ids)
    
    assert len(emitted) >= 1
    assert isinstance(accepted_count, int)
    assert all_acc is False  # Because random token won't match Qwen's true prediction
    assert speculative_engine.total_target_forward_passes > 0


@pytest.mark.asyncio
async def test_speculative_engine_generate_stream():
    from server.app import speculative_engine
    
    prompt = "The key advantages of local LLMs are"
    tokens = []
    async for tok in speculative_engine.generate_stream(prompt, max_tokens=10):
        tokens.append(tok)
        
    assert len(tokens) > 0
    assert len(tokens) <= 12
    telemetry = speculative_engine.get_telemetry()
    assert telemetry["total_tokens_generated"] > 0
    assert "acceptance_rate" in telemetry
