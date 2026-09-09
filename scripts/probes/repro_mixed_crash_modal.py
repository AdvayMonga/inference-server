"""Reproduce the mixed-workload scheduler crash on GPU, traceback forced to stdout.

The deployed app swallows the scheduler worker's exception. Here we configure logging and
drive the real scheduler + custom-cuda backend with CONTINUOUS concurrent load using real
tokenized prompts from the mixed bins (what the load-test sweep sends). `modal run` shows
stdout inline, so the worker's `logger.exception` traceback surfaces. Run:

    venv/bin/modal run scripts/probes/repro_mixed_crash_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({"BACKEND": "custom-cuda", "CUSTOM_BACKEND_BLOCKS": "1536", "MAX_ACTIVE_KV_TOKENS": "22000"})
    .add_local_python_source("inference_server")
)
app = modal.App("repro-mixed-crash", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

SEED = ("Explain how a computer works in simple terms. Walk through the CPU, memory, "
        "disk, and how they coordinate. Use plain language and concrete examples a "
        "high schooler could follow without prior background knowledge whatsoever. ")
BINS = [(0.70, 60, 100), (0.20, 250, 300), (0.10, 800, 600)]


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=400)
def run():
    import asyncio
    import logging
    import random
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from inference_server.backends import create_backend
    from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest, QueueFullError
    from inference_server.tokenizer import Tokenizer

    backend = create_backend("custom-cuda")
    backend.load_model("google/gemma-4-E2B-it")
    tokenizer = Tokenizer("google/gemma-4-E2B-it", 8192)  # mirror the server's exact path

    def build(target_tokens):
        target_chars = target_tokens * 4
        n = max(1, (target_chars // len(SEED)) + 1)
        return (SEED * n)[:target_chars]

    rng = random.Random(0)

    def draw():
        r = rng.random()
        cum = 0.0
        for w, p, mx in BINS:
            cum += w
            if r <= cum:
                return build(p), mx
        return build(BINS[-1][1]), BINS[-1][2]

    async def main():
        sched = ContinuousBatchScheduler(backend, max_batch_size=32, max_active_kv_tokens=22000)
        sched.start()
        loop = asyncio.get_running_loop()
        N = 8
        done = {"ok": 0, "err": 0}
        stop_at = 30.0  # seconds of continuous load
        t0 = loop.time()

        async def stream_one(wid, i):
            """Mirror the server: enqueue with a token_queue, drain it (decode_token each)."""
            prompt, mx = draw()
            ids = tokenizer.encode_chat(prompt, False)
            q: asyncio.Queue = asyncio.Queue()
            req = ScheduledRequest(token_ids=ids, max_tokens=mx, session_id=f"w{wid}-{i}",
                                   future=loop.create_future(), token_queue=q)
            sched.enqueue(req)
            while True:
                tok_id = await q.get()
                if tok_id is None:
                    break
                tokenizer.decode_token(tok_id)
            if req.future.done() and req.future.exception():
                raise req.future.exception()
            await req.future

        async def worker(wid):
            i = 0
            while loop.time() - t0 < stop_at:
                i += 1
                try:
                    await stream_one(wid, i)
                    done["ok"] += 1
                except QueueFullError:
                    done["err"] += 1
                except Exception as e:
                    done["err"] += 1
                    print(f"REQUEST EXCEPTION w{wid}: {type(e).__name__}: {e}")

        await asyncio.gather(*(worker(w) for w in range(N)))
        print(f"RESULT ok={done['ok']} err={done['err']} active={len(sched._active)} "
              f"pending={sched._pending_count}")
        await sched.stop()

    asyncio.run(main())


@app.local_entrypoint()
def main():
    run.remote()
