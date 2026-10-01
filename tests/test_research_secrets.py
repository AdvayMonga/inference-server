"""Credentials never reach a record: everything an agent writes is redacted first."""

from __future__ import annotations

from inference_server.research.agents.call import AgentReply, AgentSpec, call_agent
from inference_server.research.safety import secrets
from inference_server.research.safety.secrets import REDACTED, redact


def test_redaction_reaches_nested_output():
    out = redact({"a": ["sk-secret-123 here", {"b": "x sk-secret-123"}], "n": 3}, ["sk-secret-123"])
    assert out == {"a": [f"{REDACTED} here", {"b": f"x {REDACTED}"}], "n": 3}


def test_known_secrets_include_the_environment(monkeypatch):
    monkeypatch.setattr(secrets, "model_token", lambda: "tok-abcdefgh")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-abcdefgh")
    assert {"tok-abcdefgh", "rp-abcdefgh"} <= set(secrets.known_secrets())


def test_an_agent_that_prints_a_secret_has_it_redacted(tmp_path, monkeypatch):
    import inference_server.research.agents.call as call

    async def leaky(spec):
        return AgentReply({"note": "token is tok-abcdefgh"}, 0.0, 1, None, text="tok-abcdefgh")

    monkeypatch.setattr(call, "_run", leaky)
    monkeypatch.setattr(call, "known_secrets", lambda: ["tok-abcdefgh"])
    spec = AgentSpec("x", "", "", tmp_path, tmp_path / "s", None, 1, 1.0, 5.0)
    reply = call_agent(spec)
    assert reply.output == {"note": f"token is {REDACTED}"} and reply.text == REDACTED
