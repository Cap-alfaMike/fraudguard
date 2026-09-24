"""Teste de carga do /predict com concorrência (httpx assíncrono).

Uso: python scripts/benchmark.py --url http://localhost:8000 --requests 5000 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import numpy as np


def payload(rng, i):
    return {
        "transaction_id": f"bench-{i}",
        "Time": float(rng.uniform(0, 172_800)),
        "Amount": float(rng.lognormal(3, 1.3)),
        **{f"V{k}": float(rng.normal()) for k in range(1, 29)},
    }


async def run(url: str, n: int, concurrency: int) -> dict:
    rng = np.random.default_rng(0)
    bodies = [payload(rng, i) for i in range(n)]
    client_lat, server_lat, errors = [], [], 0
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=url, timeout=10) as client:
        for b in bodies[:50]:  # aquecimento
            await client.post("/predict", json=b)

        async def one(b):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await client.post("/predict", json=b)
                client_lat.append((time.perf_counter() - t0) * 1000)
                if r.status_code != 200:
                    errors += 1
                else:
                    server_lat.append(r.json()["latency_ms"])

        t0 = time.perf_counter()
        await asyncio.gather(*(one(b) for b in bodies))
        elapsed = time.perf_counter() - t0
    pct = lambda a: {f"p{q}": round(float(np.percentile(a, q)), 3) for q in (50, 95, 99)}
    return {
        "requests": n,
        "concurrency": concurrency,
        "errors": errors,
        "throughput_rps": round(n / elapsed, 1),
        "throughput_per_minute": int(n / elapsed * 60),
        "client_e2e_latency_ms": pct(client_lat),
        "server_inference_latency_ms": pct(server_lat),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a.url, a.requests, a.concurrency)), indent=2))
