"""A paged-style backend that does no model work, for tests of the scheduler's instrumentation."""

from __future__ import annotations

import threading

from inference_server.backends.base import InferenceBackend


class StubBackend(InferenceBackend):
    """Prefill returns a fake KV of the prompt length; decode adds one to every token."""

    needs_attention_mask = False

    def __init__(self):
        self._lock = threading.Lock()
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0

    def load_model(self, model_name): pass
    def generate(self, *a, **k): return []
    def generate_batch(self, *a, **k): return []
    def generate_step(self, *a, **k): return 0, None
    def stream(self, *a, **k):
        if False:
            yield 0

    def prefill(self, token_ids, session_id="default"):
        return len(token_ids), 1, len(token_ids)

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids,
                            sampling_per_row=None):
        return current_tokens.squeeze(-1) + 1, batched_kv

    def splice_into_batched(self, batched_kv, new_kv, new_kv_len):
        return [new_kv] if batched_kv is None else batched_kv + [new_kv]

    def remove_row_from_cache(self, batched_kv, row_idx):
        return batched_kv[:row_idx] + batched_kv[row_idx + 1:]

    def kv_length(self, kv): return max(kv) if kv else 0
    def is_eos(self, token_id): return False

    @property
    def device_str(self): return "cpu"
