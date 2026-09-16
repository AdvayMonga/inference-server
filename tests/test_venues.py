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
    reap_pods,
    run_instrument,
    wait_until_ready,
)


class FakeAPI:
    """Records calls; hands back whatever shape the test needs."""

    def __init__(self, *, ready_after: float = 1, create_status: int = 201,
                 terminate_status: int = 200, pods: list[dict] | None = None):
        self.calls: list[tuple[str, str]] = []
        self.ready_after, self.create_status = ready_after, create_status
        self.terminate_status = terminate_status
        self.pods = pods or []                      # what GET /pods lists
        self._gets = 0

    def __call__(self, method, path, body, key):
        self.calls.append((method, path))
        if method == "POST":
            if self.create_status >= 400:
                return self.create_status, {"error": "insufficient funds"}
            return 201, {"id": "pod-1", "costPerHr": 1.29}
        if method == "GET" and path == "/pods":
            return 200, self.pods
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

    @property
    def deleted_ids(self) -> list[str]:
        return [p.rsplit("/", 1)[1] for m, p in self.calls if m == "DELETE"]


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


def test_the_run_timeout_is_the_callers_choice_and_reaches_the_pod():
    """A run killed by this prints no closing marker, so the whole rental yields nothing. It was
    unreachable from run_instrument: every caller silently got one hour."""
    api, seen = FakeAPI(), []

    def spy(cmd):
        seen.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, emit_payload({"ok": 1}), "")

    run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api), shell=spy,
                   log=lambda _: None, run_timeout_s=5400)
    assert "timeout 5400 python scripts/x.py" in seen[-1]


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


# ---------------------------------------------------------------- the live API's own rules

def test_every_api_call_sends_a_user_agent():
    """RunPod's edge answers urllib's default User-Agent with 403 "error code: 1010" on every
    path, valid key included. Found against the live API; without this header nothing works."""
    seen: list[dict] = []

    class FakeResp:
        status = 200
        def read(self): return b"{}"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        seen.append(dict(req.headers))
        return FakeResp()

    import urllib.request

    from inference_server.research import venues
    orig = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        venues._http("GET", "/pods", None, "key-123")
    finally:
        urllib.request.urlopen = orig

    # urllib title-cases header names when it stores them.
    assert seen[0].get("User-agent") == venues.USER_AGENT


# ---------------------------------------------------------------- provisioning

def test_dependencies_are_installed_before_the_instrument_runs():
    """The image ships torch and nothing else the engine imports. Install has to happen after
    the rsync (there is no pyproject.toml before it) and before the run."""
    api, seen = FakeAPI(), []

    def spy(cmd):
        seen.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, emit_payload({"ok": 1}), "")

    run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api), shell=spy,
                   log=lambda _: None)

    steps = [i for i, c in enumerate(seen) if "rsync" in c or "pip install" in c or "python " in c]
    kinds = ["rsync" if "rsync" in seen[i] else "pip" if "pip install" in seen[i] else "run"
             for i in steps]
    assert kinds == ["rsync", "pip", "run"], kinds
    assert "-e ." in seen[steps[1]], "must install the synced tree, not a published wheel"


def test_a_failed_install_terminates_the_pod_and_never_runs_the_instrument():
    """Renting a GPU and then failing to install is the cheapest way to waste money slowly."""
    api, seen = FakeAPI(), []

    def spy(cmd):
        joined = " ".join(cmd)
        seen.append(joined)
        rc = 1 if "pip install" in joined else 0
        return subprocess.CompletedProcess(cmd, rc, "No matching distribution", "")

    with pytest.raises(VenueError, match="pip install failed"):
        run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api), shell=spy,
                       log=lambda _: None)

    assert api.terminated
    assert not any("python scripts/x.py" in c for c in seen)


# ---------------------------------------------------------------- the first real instrument

def test_venue_smoke_emits_a_payload_the_venue_can_parse():
    """The instrument side of the contract, exercised for real: run the smoke script as a
    subprocess and parse its stdout exactly as run_instrument would."""
    import os
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src"),
           "RESEARCH_ENGINE_SHA": "abc123", "RESEARCH_RUN_GROUP": "grp-7"}
    r = subprocess.run([sys.executable, str(repo / "scripts" / "tools" / "venue_smoke.py")],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]

    payload = extract_payload(r.stdout)
    assert payload["panels"] == [], "a smoke run measures no engine behaviour, so it emits none"
    smoke = payload["smoke"]
    assert smoke["provenance"] == {"RESEARCH_ENGINE_SHA": "abc123", "RESEARCH_RUN_GROUP": "grp-7"}
    for mod, ver in smoke["imports"].items():
        assert not ver.startswith("MISSING"), f"{mod} is not importable: {ver}"
    assert "available" in smoke["gpu"]


# ---------------------------------------------------------------- a gate's verdict

def test_a_gate_that_reports_then_fails_still_brings_its_verdict_home():
    """cuda_gate.py exits 1 when a check fails, so a direct run needs no parser. Through the
    venue that exit must read as a verdict, not a broken run — and still terminate the pod."""
    api = FakeAPI()

    def verdict(cmd):
        rc = 1 if "python " in " ".join(cmd) else 0
        return subprocess.CompletedProcess(cmd, rc, emit_payload({"gate": {"passed": False}}), "")

    out = run_instrument("scripts/gpu_tests/cuda_gate.py", {}, repo="/repo",
                         client=_client(api), shell=verdict, log=lambda _: None)
    assert out == {"gate": {"passed": False}}
    assert api.terminated


def _exit_nonzero_with(stdout: str):
    def run(cmd):
        rc = 1 if "python " in " ".join(cmd) else 0
        return subprocess.CompletedProcess(cmd, rc, stdout, "Killed")
    return run


def test_an_instrument_that_reports_its_own_failure_is_heard():
    """PR #26's replay instrument emits {"error", "log_tail"} when the server never comes up
    and exits 1. That report must reach the launcher, not be flattened into a VenueError."""
    api = FakeAPI()
    report = {"error": "server not ready after 120s", "log_tail": ["loading weights", "OOM"]}
    out = run_instrument("scripts/bench/replay.py", {}, repo="/repo", client=_client(api),
                         shell=_exit_nonzero_with(emit_payload(report)), log=lambda _: None)
    assert out == report
    assert api.terminated


def test_a_partial_run_keeps_the_panels_it_did_measure_alongside_its_error():
    """The replay instrument's other honest exit: two of three configs finished, so the payload
    carries their panels AND an `error` naming the third. Those GPU-minutes are already billed —
    dropping them on the exit code would make a partly-failed rental worth nothing. `error` is
    what distinguishes this from the panels-only crash below."""
    api, logs = FakeAPI(), []
    partial = {"panels": [{"x": 1}, {"x": 2}], "replays": [], "error": "1 replay(s) failed: x4"}
    out = run_instrument("scripts/bench/replay.py", {}, repo="/repo", client=_client(api),
                         shell=_exit_nonzero_with(emit_payload(partial)), log=logs.append)
    assert out == partial
    assert any("reported why — keeping its payload" in ln for ln in logs)
    assert api.terminated


def test_a_nonzero_exit_with_only_panels_is_still_a_broken_run():
    """A measurement that printed panels and then crashed is not evidence."""
    api = FakeAPI()
    with pytest.raises(VenueError, match="exited 1"):
        run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api),
                       shell=_exit_nonzero_with(emit_payload({"panels": [{"x": 1}]})),
                       log=lambda _: None)
    assert api.terminated


def test_a_nonzero_exit_with_no_payload_is_still_a_broken_run():
    api = FakeAPI()
    with pytest.raises(VenueError, match="exited 1"):
        run_instrument("scripts/x.py", {}, repo="/repo", client=_client(api),
                       shell=_exit_nonzero_with("<<<RESEARCH_PANEL_JSON\n{"), log=lambda _: None)
    assert api.terminated


# ---------------------------------------------------------------- the reaper

def _pods():
    return [
        {"id": "ours-new", "name": "inference-server-instrument",
         "lastStartedAt": "2026-09-16T07:05:00.000Z", "costPerHr": 0.44},
        {"id": "ours-old", "name": "inference-server-instrument",
         "lastStartedAt": "2026-09-15T22:00:00.000Z", "costPerHr": 0.44},
        {"id": "ours-undated", "name": "inference-server-instrument", "costPerHr": 0.44},
        {"id": "theirs", "name": "someone-elses-training-run",
         "lastStartedAt": "2026-09-16T07:06:00.000Z", "costPerHr": 2.99},
    ]


def test_list_reads_names_and_start_times():
    api = FakeAPI(pods=_pods())
    pods = _client(api).list()
    assert [(p.id, p.name, p.last_started_at) for p in pods][:2] == [
        ("ours-new", "inference-server-instrument", "2026-09-16T07:05:00.000Z"),
        ("ours-old", "inference-server-instrument", "2026-09-15T22:00:00.000Z"),
    ]
    assert api.calls == [("GET", "/pods")]


def test_reaper_never_terminates_a_pod_it_did_not_name():
    """The whole safety property. A shared account can hold someone's real job."""
    api = FakeAPI(pods=_pods())
    reaped = reap_pods(_client(api), log=lambda _: None)
    assert "theirs" not in reaped and "theirs" not in api.deleted_ids
    assert set(reaped) == {"ours-new", "ours-old", "ours-undated"}


def test_reaper_with_since_leaves_pods_started_before_the_job():
    """`since` is the job start: a local run that began earlier with the default name survives.
    An undated pod carries our name, so it goes."""
    api = FakeAPI(pods=_pods())
    logs: list[str] = []
    reaped = reap_pods(_client(api), since="2026-09-16T07:00:00Z", log=logs.append)
    assert set(reaped) == {"ours-new", "ours-undated"}
    assert api.deleted_ids == reaped
    assert any("leaving ours-old" in ln for ln in logs)


def test_reaper_with_nothing_to_reap_calls_no_delete():
    api = FakeAPI(pods=[_pods()[3]])
    assert reap_pods(_client(api), log=lambda _: None) == []
    assert not api.terminated


def test_reaper_cli_without_a_key_rents_and_kills_nothing():
    """CI runs the reaper with `if: always()`, secrets or not. No key means nothing was rented."""
    import os
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}
    r = subprocess.run([sys.executable, str(repo / "scripts" / "tools" / "runpod_reap.py"),
                        "--since", "2026-09-16T07:00:00Z"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr[-500:]
    assert "nothing to reap" in r.stdout



# ---------------------------------------------------------------- the API lying to the reaper

def test_a_malformed_row_is_skipped_and_the_sweep_continues():
    """One bad record must not abort the sweep: every other leaked pod would keep billing."""
    rows = [{"name": "inference-server-instrument", "costPerHr": 0.44},        # no id
            "not even a dict",
            {"id": "ours-1", "name": "inference-server-instrument", "costPerHr": "n/a"},
            {"id": "ours-2", "name": "inference-server-instrument", "lastStartedAt": 12345}]
    api, logs = FakeAPI(pods=rows), []
    pods = _client(api).list(log=logs.append)
    assert [p.id for p in pods] == ["ours-1", "ours-2"]
    assert pods[0].cost_per_hr == 0.0 and pods[1].last_started_at == "12345"
    assert sum("malformed" in ln for ln in logs) == 2

    reaped = reap_pods(_client(api), since="2026-09-16T07:00:00Z", log=logs.append)
    assert set(reaped) == {"ours-1", "ours-2"}


def test_an_unreadable_start_time_on_our_pod_means_terminate_not_crash():
    rows = [{"id": "ours-bad-ts", "name": "inference-server-instrument",
             "lastStartedAt": "yesterday-ish"},
            {"id": "ours-old", "name": "inference-server-instrument",
             "lastStartedAt": "2026-09-15T22:00:00Z"}]
    api, logs = FakeAPI(pods=rows), []
    reaped = reap_pods(_client(api), since="2026-09-16T07:00:00Z", log=logs.append)
    assert reaped == ["ours-bad-ts"]
    assert any("no readable start time" in ln for ln in logs)


def test_an_unexpected_pods_response_shape_is_logged_not_read_as_empty():
    api, logs = FakeAPI(pods={"data": [{"id": "hidden"}]}), []   # dict without a "pods" key
    assert _client(api).list(log=logs.append) == []
    assert any("unexpected shape" in ln for ln in logs)
    api2 = FakeAPI(pods={"pods": [{"id": "wrapped", "name": "x"}]})
    assert [p.id for p in api2 and _client(api2).list(log=logs.append)] == ["wrapped"]


def test_a_bad_since_is_refused_before_anything_is_touched():
    api = FakeAPI(pods=_pods())
    with pytest.raises(VenueError, match="ISO-8601"):
        reap_pods(_client(api), since="last monday", log=lambda _: None)
    assert not api.terminated
