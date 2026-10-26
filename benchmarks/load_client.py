import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Tuple

BASE_URL = "http://127.0.0.1:8002/v1/completions"

async def send_single_request(
    session: aiohttp.ClientSession, 
    prompt: str, 
    max_tokens: int = 25, 
    mode: str = "batching"
) -> Dict[str, float]:
    """
    Sends a single streaming request and tracks TTFT and total duration.
    """
    start = time.time()
    first_token_time = None
    token_chunks = 0
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "mode": mode
    }
    
    try:
        async with session.post(BASE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
            if response.status != 200:
                print(f"[Error] HTTP {response.status}")
                return {"tokens": 0, "ttft": 0, "duration": 0}
                
            async for chunk in response.content.iter_any():
                if chunk:
                    if first_token_time is None:
                        first_token_time = time.time() - start
                    token_chunks += 1
    except Exception as e:
        print(f"[Exception] Request failed: {e}")
        return {"tokens": 0, "ttft": 0, "duration": 0}

    duration = time.time() - start
    return {
        "tokens": token_chunks,
        "ttft": first_token_time or duration,
        "duration": duration
    }

async def benchmark_run(mode: str, concurrency: int, max_tokens: int = 20) -> Dict[str, float]:
    """
    Runs a batch of concurrent requests and aggregates throughput and latency metrics.
    """
    print(f"\n[Benchmarking] Mode='{mode}' | Concurrency={concurrency} | MaxTokens={max_tokens}...")
    prompts = [
        f"Explain concept {i}: What is gradient descent in machine learning?" 
        for i in range(concurrency)
    ]
    
    async with aiohttp.ClientSession() as session:
        wall_start = time.time()
        tasks = [
            send_single_request(session, prompt, max_tokens=max_tokens, mode=mode)
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.time() - wall_start
        
    valid_results = [r for r in results if r["tokens"] > 0]
    total_tokens = sum(r["tokens"] for r in valid_results)
    mean_ttft = sum(r["ttft"] for r in valid_results) / len(valid_results) if valid_results else 0
    throughput = total_tokens / wall_time if wall_time > 0 else 0
    
    print(f"  Wall Time:   {wall_time:.2f}s")
    print(f"  Total Toks:  {total_tokens}")
    print(f"  Mean TTFT:   {mean_ttft:.3f}s")
    print(f"  Throughput:  {throughput:.2f} tokens/sec")
    
    return {
        "mode": mode,
        "concurrency": concurrency,
        "wall_time_sec": round(wall_time, 2),
        "total_tokens": total_tokens,
        "mean_ttft_sec": round(mean_ttft, 3),
        "throughput_tps": round(throughput, 2)
    }

async def main():
    print("=" * 60)
    print("   INFERENCE LAB: M1 vs M2 CONTINUOUS BATCHING BENCHMARK")
    print("=" * 60)
    
    # 1. Baseline Sequential (Mode='real', 4 requests sent one by one)
    print("\n--- Phase 1: Baseline Sequential (Concurrency = 1) ---")
    sequential_start = time.time()
    seq_results = []
    async with aiohttp.ClientSession() as session:
        for i in range(4):
            res = await send_single_request(session, f"Question {i}: Define AI.", max_tokens=15, mode="real")
            seq_results.append(res)
    seq_wall = time.time() - sequential_start
    seq_tokens = sum(r["tokens"] for r in seq_results)
    seq_tps = seq_tokens / seq_wall if seq_wall > 0 else 0
    seq_ttft = sum(r["ttft"] for r in seq_results) / len(seq_results) if seq_results else 0
    
    m1_data = {
        "mode": "M1 Sequential (Baseline)",
        "concurrency": 1,
        "wall_time_sec": round(seq_wall, 2),
        "total_tokens": seq_tokens,
        "mean_ttft_sec": round(seq_ttft, 3),
        "throughput_tps": round(seq_tps, 2)
    }
    print(f"  Wall Time: {seq_wall:.2f}s | Throughput: {seq_tps:.2f} tok/s | TTFT: {seq_ttft:.3f}s")
    
    # 2. Continuous Batching (Mode='batching', 2 concurrent requests)
    print("\n--- Phase 2: Continuous Batching (Concurrency = 2) ---")
    m2_data_2 = await benchmark_run(mode="batching", concurrency=2, max_tokens=15)
    
    # 3. Continuous Batching (Mode='batching', 4 concurrent requests)
    print("\n--- Phase 3: Continuous Batching (Concurrency = 4) ---")
    m2_data_4 = await benchmark_run(mode="batching", concurrency=4, max_tokens=15)
    
    # Summary Table
    print("\n" + "=" * 65)
    print("               FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 65)
    print(f"{'Mode':<28} | {'Conc':<5} | {'Wall (s)':<9} | {'TTFT (s)':<9} | {'Tok/s':<8}")
    print("-" * 65)
    for row in [m1_data, m2_data_2, m2_data_4]:
        print(f"{row['mode']:<28} | {row['concurrency']:<5} | {row['wall_time_sec']:<9} | {row['mean_ttft_sec']:<9} | {row['throughput_tps']:<8}")
    print("=" * 65)
    
    # Save to disk
    with open("benchmarks/benchmark_results.json", "w") as f:
        json.dump([m1_data, m2_data_2, m2_data_4], f, indent=2)
    print("[Success] Results saved to benchmarks/benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())
