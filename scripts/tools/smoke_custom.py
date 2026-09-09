"""Local smoke: real server startup path with the custom backend, over HTTP.

Drives the actual lifespan (create_backend(settings.backend) → CustomTorchBackend,
CacheManager, ContinuousBatchScheduler) and a /generate round-trip. MPS exercises
device placement the CPU unit tests can't. Set BACKEND before running, e.g.:

    BACKEND=custom-mps venv/bin/python scripts/tools/smoke_custom.py
"""

import asyncio
import os

os.environ.setdefault("BACKEND", "custom-mps")
os.environ.setdefault("CUSTOM_BACKEND_BLOCKS", "128")
os.environ.setdefault("CUSTOM_BACKEND_BLOCK_SIZE", "16")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from inference_server.config import settings  # noqa: E402
from inference_server.server import app, lifespan  # noqa: E402


async def main():
    print(f"BACKEND resolves to: {settings.backend}")
    async with lifespan(app):  # real startup: load model, build scheduler
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test", timeout=120) as c:
            for ep in ("/health", "/ready"):
                r = await c.get(ep)
                print(f"GET {ep} -> {r.status_code} {r.text}")

            r = await c.post("/generate", json={"text": "The capital of France is", "max_tokens": 8})
            print(f"POST /generate -> {r.status_code}")
            print(r.json() if r.status_code == 200 else r.text)

            # Second identical call should hit the prefix cache.
            r2 = await c.post("/generate", json={"text": "The capital of France is", "max_tokens": 8})
            print(f"POST /generate (repeat) -> {r2.status_code}")
            print(r2.json() if r2.status_code == 200 else r2.text)

            stats = (await c.get("/scheduler/stats")).json()
            print("scheduler stats:", {k: stats[k] for k in ("total_completed", "total_rejected", "active_size") if k in stats})


if __name__ == "__main__":
    asyncio.run(main())
