"""A venue that can rent hardware can also forget to give it back.

Every test here is about one of two failure classes:

  money  — a pod that outlives the run bills until someone notices it in a web console.
  evidence — a run that produced no parseable panel must fail loudly, because the alternative is
             a silent empty result that looks like "no effect" and gets believed.
"""

from __future__ import annotations

import subprocess

import pytest

from inference_server.research.venues import (
    PodSpec,
    RunPodClient,
    VenueError,
    emit_payload,
    extract_payload,
    run_instrument,
    wait_until_ready,
)


class FakeAPI:
    """Records calls; hands back whatever shape the test needs."""

    def __init__(self, *, ready_after: float = 1, create_status: int = 201,
                 terminate_status: int = 200):
        self.calls: list[tuple[str, str]] = []
        self.ready_after, self.create_status = ready_after, create_status
        self.terminate_status = terminate_status
        self._gets = 0

    def __call__(self, method, path, body, key):
        self.calls.append((method, path))
        if method == "POST":
            if self.create_status >= 400:
                return self.create_status, {"error": "insufficient funds"}
            return 201, {"id": "pod-1", "costPerHr": 1.29}
        if method == "GET":
            self._gets += 1
            if self._gets < self.ready_after:
                return 200, {"id": "pod-1", "publicIp": None, "portMappings": []}
            return 200, {"id": "pod-1", "publicIp": "1.2.3.4",
                         "portMappings": [{"privatePort": 22, "publicPort": 40123}],
                         "costPerHr": 1.29}
        return self.terminate_status, {} if self.terminate_status < 400 else {"error": "nope"}

    @property
    def terminated(self) -> bool:
        return any(m == "DELETE" for m, _ in self.calls)


def _client(api: FakeAPI) -> RunPodClient:
    return RunPodClient(api_key="test-key", transport=api)


def _shell_ok(stdout: str):
    def run(cmd):
        return subprocess.CompletedProcess(cmd, 0, stdout, "")
    return run


# ---------------------------------------------------------------- money

def test_the_pod_is_terminated_when_the_instrument_succeeds():
    api = FakeAPI()
    out = run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api),
                         shell=_shell_ok(emit_payload({"rows": [1]})), log=lambda _: None)

    assert out == {"rows": [1]}
    assert api.terminated


def test_the_pod_is_terminated_when_the_instrument_crashes():
    """The expensive case. A benchmark that OOMs must not leave an A100 running."""
    api = FakeAPI()

    def boom(cmd):
        if cmd[0] == "rsync":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "CUDA out of memory")

    with pytest.raises(VenueError, match="CUDA out of memory"):
        run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api),
                       shell=boom, log=lambda _: None)
    assert api.terminated, "a crashed run still rented a GPU"


def test_the_pod_is_terminated_when_it_never_becomes_reachable():
    """Timing out waiting for ssh is the sneakiest leak: nothing ran, so nothing feels owed."""
    api = FakeAPI(ready_after=float("inf"))   # never reachable; a finite count is not the
                                          # same thing when sleep is a no-op

    with pytest.raises(VenueError, match="no ssh endpoint"):
        run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api),
                       shell=_shell_ok(""), log=lambda _: None,
                       ready_timeout_s=0.01, sleep=lambda _: None)
    assert api.terminated


def test_a_failed_teardown_does_not_hide_the_real_error():
    """If terminate raised, the traceback would blame the teardown and the actual failure would
    never be seen — while the pod is still billing either way."""
    api = FakeAPI(terminate_status=500)
    logs: list[str] = []

    def boom(cmd):
        if cmd[0] == "rsync":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "the real failure")

    with pytest.raises(VenueError, match="the real failure"):
        run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api),
                       shell=boom, log=logs.append)
    assert any("still billing" in ln for ln in logs), "a leak must be reported, loudly"


def test_a_create_that_is_refused_rents_nothing():
    api = FakeAPI(create_status=402)
    with pytest.raises(VenueError, match="402"):
        run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api),
                       shell=_shell_ok(""), log=lambda _: None)
    assert not api.terminated, "nothing was created, so nothing should be deleted"


def test_rsync_failure_is_reported_and_terminates():
    api = FakeAPI()

    def bad_sync(cmd):
        return subprocess.CompletedProcess(cmd, 23, "", "rsync: connection unexpectedly closed")

    with pytest.raises(VenueError, match="rsync to pod failed"):
        run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api),
                       shell=bad_sync, log=lambda _: None)
    assert api.terminated


# ---------------------------------------------------------------- evidence

def test_payload_round_trips():
    payload = {"rows": [{"rate": 2, "tok_s": 780.0}], "panels": [{"validity": {"run_id": "r1"}}]}
    assert extract_payload(emit_payload(payload)) == payload


def test_payload_survives_surrounding_noise():
    """torch, HF and our own progress prints share this stdout."""
    noisy = ("Downloading shards: 100%\n" + emit_payload({"ok": 1})
             + "\n[venue] done\nSome trailing warning\n")
    assert extract_payload(noisy) == {"ok": 1}


def test_the_last_payload_wins():
    """A resumed or retried instrument prints more than one; the finished run is the last."""
    two = emit_payload({"attempt": 1}) + emit_payload({"attempt": 2})
    assert extract_payload(two) == {"attempt": 2}


def test_a_run_with_no_payload_is_an_error_not_an_empty_result():
    """The failure this guards: an empty result reads as 'no effect measured' and gets believed."""
    with pytest.raises(VenueError, match="no evidence"):
        extract_payload("loaded model\nstarted sweep\nKilled\n")


def test_a_truncated_payload_is_an_error():
    killed = "noise\n" + emit_payload({"rows": []}).split("RESEARCH_PANEL_JSON>>>")[0]
    with pytest.raises(VenueError, match="never closed"):
        extract_payload(killed)


def test_malformed_payload_json_is_an_error():
    from inference_server.research.venues import PANEL_BEGIN, PANEL_END
    with pytest.raises(VenueError, match="not valid JSON"):
        extract_payload(f"{PANEL_BEGIN}\n{{'not': 'json'}}\n{PANEL_END}")


# ---------------------------------------------------------------- plumbing

def test_ssh_port_is_read_from_the_mapping():
    api = FakeAPI()
    pod = wait_until_ready(_client(api), "pod-1", sleep=lambda _: None)
    assert (pod.host, pod.port) == ("1.2.3.4", 40123)


def test_polling_waits_for_the_ip_rather_than_failing_fast():
    """A pod takes tens of seconds to get an IP; treating the first empty read as fatal would
    make every run fail."""
    api = FakeAPI(ready_after=4)
    pod = wait_until_ready(_client(api), "pod-1", sleep=lambda _: None)
    assert pod.reachable and api._gets == 4


def test_missing_api_key_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(VenueError, match="runpod.io/console"):
        RunPodClient()


def test_spec_defaults_match_the_modal_deployment():
    """A panel measured on a pod is compared against panels measured on Modal. If the GPU or the
    disk budget differ silently, the comparison is between machines, not between code."""
    body = PodSpec().as_body()
    assert body["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"]     # modal_app.py gpu="A100-80GB"
    assert body["computeType"] == "GPU" and body["gpuCount"] == 1
    assert body["containerDiskInGb"] >= 80, "E4B weights plus a cuda torch do not fit in 50GB"
    assert "22/tcp" in body["ports"]


def test_provenance_env_reaches_the_remote_command():
    """Panels are emitted on a box with no git repo. If these do not arrive, every run is
    unattributable — the exact reason run_instrument.sh exists."""
    api, seen = FakeAPI(), []

    def spy(cmd):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, emit_payload({"ok": 1}), "")

    run_instrument("scripts/bench/x.py", {"RESEARCH_ENGINE_SHA": "abc123",
                                          "RESEARCH_RUN_GROUP": "grp-1"},
                   repo="/repo", client=_client(api), shell=spy, log=lambda _: None)

    remote = " ".join(seen[-1])
    assert "RESEARCH_ENGINE_SHA=abc123" in remote and "RESEARCH_RUN_GROUP=grp-1" in remote
    assert "PYTHONPATH=/workspace/repo/src" in remote
