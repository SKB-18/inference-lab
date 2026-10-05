import asyncio
import aiohttp
import time

async def fetch(session, prompt: str):
    start = time.time()
    tokens = 0
    async with session.post(
        "http://127.0.0.1:8000/v1/completions",
        json={"prompt": prompt, "max_tokens": 50}
    ) as response:
        async for _ in response.content.iter_any():
            # Simply counting chunks as tokens for this simulation
            tokens += 1
    
    elapsed = time.time() - start
    return tokens, elapsed

async def run_load_test(concurrency: int):
    print(f"Starting load test with {concurrency} concurrent requests...")
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        tasks = [fetch(session, f"Request {i}") for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        total_tokens = sum(r[0] for r in results)
        
        print(f"--- Results for Concurrency {concurrency} ---")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Total Tokens: {total_tokens}")
        print(f"Throughput: {total_tokens / total_time:.2f} tokens/sec")

if __name__ == "__main__":
    # Test with 20 concurrent requests
    asyncio.run(run_load_test(20))
