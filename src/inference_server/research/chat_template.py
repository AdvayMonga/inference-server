"""What the tokenizer turns a corpus prompt into — the variable chat templating introduces.

Posting a corpus prompt to `/v1/chat/completions` instead of `/v1/completions` is what makes an
instruct model answer at all (`kb-20260919-6ef4e6bf`). The cost is that the tokens the model sees
are no longer the trace's bytes: they are the trace's bytes wrapped in whatever chat template the
tokenizer happens to ship, with whatever `enable_thinking` the shim happens to pass. A template
revision would then move every number in the panel with nothing in the record changing — a
silent variable swapped for a silent variable.

So the fingerprint rides in the validity block beside `corpus_version` and `device_state`, and
`compare.py` refuses across it.

`verify()` closes the last gap. The fingerprint is computed by the CLIENT; the SERVER is what
actually applies the template. They are the same tokenizer on this box and on the pod (the
replayer runs beside the engine in both shapes), but nothing enforces that, so the rendered
length is checked against the `prompt_tokens` the server reported for a request whose text we
know. `verified=False` in a panel means the stamp describes a different tokenizer than the one
that served the run.

transformers is imported lazily and every failure yields None: the loop's CI lane installs
stdlib only, and an unknown fingerprint must be an ordinary outcome, never an exception. Same
contract as `harness.device_state_from_env()`.
"""

from __future__ import annotations

import hashlib
from typing import Any

# What openai_shim.chat_completions passes to Tokenizer.encode_messages. Recorded rather than
# assumed: it decides whether the template injects a `<|think|>` system turn, which on
# gemma-4-E2B-it is the difference between a corpus that terminates early and one where every
# request runs to max_tokens. False since the shim stopped inheriting encode_messages' True
# default (kb-20260919-9ea56f98); panels either side of that are not comparable, and the
# fingerprint is what makes compare.py say so.
SHIM_ENABLE_THINKING = False

_CACHE: dict[str, Any] = {}


def _tokenizer(model_name: str):
    """AutoTokenizer, cached per process. None when transformers or the model is unavailable."""
    if model_name not in _CACHE:
        try:
            from transformers import AutoTokenizer
            _CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
        except Exception:                # noqa: BLE001 — no tokenizer is a state, not a crash
            _CACHE[model_name] = None
    return _CACHE[model_name]


def rendered_len(model_name: str, prompt: str, *,
                 enable_thinking: bool = SHIM_ENABLE_THINKING) -> int | None:
    """Token count the shim's chat route would encode `prompt` to, or None if unknowable."""
    tk = _tokenizer(model_name)
    if tk is None or not getattr(tk, "chat_template", None):
        return None
    try:
        return len(tk.apply_chat_template(
            [{"role": "user", "content": prompt}], return_dict=True, return_tensors=None,
            add_generation_prompt=True, enable_thinking=enable_thinking)["input_ids"])
    except Exception:                    # noqa: BLE001
        return None


def fingerprint(model_name: str, *,
                enable_thinking: bool = SHIM_ENABLE_THINKING) -> dict[str, Any] | None:
    """Identify the template that will be applied. None when it cannot be read.

    `probe_tokens` is the rendered length of a fixed one-word prompt. It catches a tokenizer or
    transformers upgrade that renders the same template string differently — the hash alone
    would not.
    """
    tk = _tokenizer(model_name)
    template = getattr(tk, "chat_template", None) if tk is not None else None
    if not isinstance(template, str):
        return None
    return {
        "tokenizer": model_name,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "enable_thinking": enable_thinking,
        "probe_tokens": rendered_len(model_name, "probe", enable_thinking=enable_thinking),
        "verified": None,
    }


def verify(fp: dict[str, Any] | None, prompt: str, server_prompt_tokens: int | None,
           *, enable_thinking: bool = SHIM_ENABLE_THINKING) -> dict[str, Any] | None:
    """Set `verified` by re-rendering `prompt` and comparing with what the server encoded.

    Left None when either side is unknown — "not checked" and "checked and wrong" are different
    facts, and only the second one invalidates the panel.
    """
    if fp is None or server_prompt_tokens is None:
        return fp
    mine = rendered_len(fp["tokenizer"], prompt, enable_thinking=enable_thinking)
    if mine is None:
        return fp
    fp["verified"] = mine == server_prompt_tokens
    if not fp["verified"]:
        fp["verify_detail"] = (f"client rendered {mine} prompt tokens, server encoded "
                               f"{server_prompt_tokens}")
    return fp
