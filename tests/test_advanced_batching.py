import pytest
import asyncio
import torch
from engine.batching_queue import BatchingEngine, BatchedRequest


@pytest.fixture
def dummy_batching_engine():
    # Lightweight mock model/tokenizer to test batch scheduling mechanics fast
    class MockTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token = "<eos>"
        eos_token_id = 2
        padding_side = "left"
        chat_template = None

        def encode(self, text, add_special_tokens=True):
            return [10, 20, 30]

        def decode(self, token_ids):
            return f"tok_{token_ids[0]}"

    class MockModel:
        def __call__(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            # Return logits where argmax predicts token 99
            logits = torch.zeros(batch_size, seq_len, 100)
            logits[:, -1, 99] = 10.0
            class Output:
                pass
            out = Output()
            out.logits = logits
            return out

        def eval(self):
            pass

    return BatchingEngine(
        max_batch_size=4,
        model=MockModel(),
        tokenizer=MockTokenizer(),
        device="cpu"
    )


def test_batching_step_batch_left_padding(dummy_batching_engine):
    # Two requests with different prompt lengths
    req1 = BatchedRequest(
        request_id="r1",
        prompt="short",
        input_ids=[1, 2],
        max_tokens=5
    )
    req2 = BatchedRequest(
        request_id="r2",
        prompt="longer prompt",
        input_ids=[3, 4, 5, 6],
        max_tokens=5
    )
    
    active_batch = [req1, req2]
    next_batch = dummy_batching_engine._step_batch(active_batch)
    
    assert len(next_batch) == 2
    # Both generated 1 token and appended to input_ids
    assert req1.tokens_generated == 1
    assert req1.input_ids[-1] == 99
    assert req2.tokens_generated == 1
    assert req2.input_ids[-1] == 99


def test_batching_dynamic_sequence_retirement(dummy_batching_engine):
    # Request 1 needs only 1 token (will retire immediately)
    req1 = BatchedRequest(
        request_id="r1",
        prompt="one token",
        input_ids=[10],
        max_tokens=1
    )
    # Request 2 needs 3 tokens (will stay in batch)
    req2 = BatchedRequest(
        request_id="r2",
        prompt="three tokens",
        input_ids=[20, 21],
        max_tokens=3
    )

    next_batch = dummy_batching_engine._step_batch([req1, req2])
    
    # req1 must have retired (not in next_batch) and received None sentinel
    assert len(next_batch) == 1
    assert next_batch[0].request_id == "r2"
    assert req1.queue.get_nowait() == "tok_99"
    assert req1.queue.get_nowait() is None  # Sentinel


def test_batching_eos_retirement(dummy_batching_engine):
    # If model emits EOS (token 2), sequence retires regardless of max_tokens
    class EosMockModel:
        def __call__(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 100)
            logits[:, -1, 2] = 10.0  # Emits EOS
            class Output:
                pass
            out = Output()
            out.logits = logits
            return out
        def eval(self): pass

    dummy_batching_engine.model = EosMockModel()
    req = BatchedRequest(request_id="r_eos", prompt="eos test", input_ids=[1], max_tokens=10)
    
    next_batch = dummy_batching_engine._step_batch([req])
    assert len(next_batch) == 0  # Retired immediately on EOS
    assert req.queue.get_nowait() == "tok_2"
    assert req.queue.get_nowait() is None


@pytest.mark.asyncio
async def test_batching_generate_stream_zero_tokens(dummy_batching_engine):
    tokens = [t async for t in dummy_batching_engine.generate_stream("hi", max_tokens=0)]
    assert tokens == []


@pytest.mark.asyncio
async def test_batching_lifecycle_start_stop(dummy_batching_engine):
    await dummy_batching_engine.start()
    assert dummy_batching_engine.running is True
    await dummy_batching_engine.stop()
    assert dummy_batching_engine.running is False
