"""Rent a GPU, run one instrument on it, bring the panel home, give the GPU back.

Modal is not a venue in this codebase, it is the ONLY way to reach a GPU: 34 instruments open
with `import modal` and end with `sweep.remote()`. `budget.py` can already route a hypothesis to
`runpod-a100`, but routing only names a price — nothing can execute there. This module is the
executor half.

The contract is deliberately the same one Modal already gives us, because that contract is what
made the instruments writable:

    ship the repo -> run one python file on a GPU -> get a JSON payload back -> keep no machine

Modal implements it with a built image and a pickled return value. A rented pod implements it
with rsync and stdout. The instrument does not need to know which it got.

What differs, and why the shapes are not identical:

  * Modal images are content-addressed and cached; a pod starts from a public docker image and
    pip-installs on every boot. Bake a template or accept ~3 minutes of setup per run.
  * A Modal function that returns is billed to the second. A pod bills until it is TERMINATED,
    so every path out of `run_instrument` terminates, including the ones that raise. An orphaned
    A100 costs about $1.30/hour forever, which is the only way this module can lose real money.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

API_BASE = "https://rest.runpod.io/v1"
API_KEY_ENV = "RUNPOD_API_KEY"

# RunPod sits behind a bot filter that refuses urllib's default User-Agent with 403 "error code:
# 1010" on EVERY path, valid key included. Found by calling the live API, not by reading docs.
USER_AGENT = "inference-server-research/1.0"

# An instrument prints its machine record between these markers. stdout is the only channel a
# rented box gives us for free, and it is shared with torch's warnings, so the payload has to be
# findable rather than assumed to be the whole stream.
PANEL_BEGIN = "<<<RESEARCH_PANEL_JSON"
PANEL_END = "RESEARCH_PANEL_JSON>>>"


class VenueError(RuntimeError):
    """A venue could not run the job. Never raised for a FAILING experiment, only a broken one."""


class InstrumentFailed(VenueError):
    """The instrument exited non-zero. Carries its stdout, because a partly-successful run can
    exit non-zero AND have printed a payload, and that payload is evidence already paid for."""

    def __init__(self, message: str, stdout: str = ""):
        super().__init__(message)
        self.stdout = stdout


# ------------------------------------------------------------------ payload transport

def extract_payload(stdout: str) -> dict[str, Any]:
    """Pull the machine record out of a run's stdout.

    Takes the LAST marked block: a retried or resumed instrument can print more than one, and the
    final one is the run that finished.
    """
    start = stdout.rfind(PANEL_BEGIN)
    if start < 0:
        raise VenueError(
            f"no {PANEL_BEGIN} block in the instrument's output — it did not reach the point "
            f"where it emits panels, so the run produced no evidence")
    body = stdout[start + len(PANEL_BEGIN):]
    end = body.find(PANEL_END)
    if end < 0:
        raise VenueError(
            f"{PANEL_BEGIN} block was never closed — the run was almost certainly killed "
            f"mid-print (OOM, timeout, or a terminated pod)")
    try:
        return json.loads(body[:end])
    except json.JSONDecodeError as e:
        raise VenueError(f"panel block is not valid JSON: {e}") from e


def reports_its_own_failure(stdout: str) -> bool:
    """True when stdout holds a parseable payload that is a verdict (`gate`) or an honest
    failure report (`error`). A `panels`-only payload does not qualify: a measurement that then
    crashed is not evidence, whatever it printed first."""
    try:
        payload = extract_payload(stdout)
    except VenueError:
        return False
    return "gate" in payload or "error" in payload


def emit_payload(payload: dict[str, Any]) -> str:
    """The instrument side of `extract_payload`. Kept here so the two cannot drift apart."""
    return f"\n{PANEL_BEGIN}\n{json.dumps(payload)}\n{PANEL_END}\n"


# ------------------------------------------------------------------ the API

@dataclass
class Pod:
    id: str
    host: str | None = None
    port: int | None = None
    cost_per_hr: float = 0.0
    name: str = ""
    last_started_at: str | None = None          # ISO-8601 from the API; the reaper's clock

    @property
    def reachable(self) -> bool:
        return bool(self.host and self.port)


def _cost(v: Any) -> float:
    """costPerHr as a float, 0.0 when the API sends nothing or something that is not a number."""
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _http(method: str, path: str, body: dict | None, key: str) -> tuple[int, dict]:
    """Default transport. urllib rather than httpx: this must work in a bare venv."""
    req = urllib.request.Request(
        f"{API_BASE}{path}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:500]}


class RunPodClient:
    """The three calls a one-shot benchmark needs. Not a general SDK.

    `transport` is injectable because the interesting failures here — a pod that never gets an IP,
    a create that 402s on an empty balance — must be testable without renting anything.
    """

    def __init__(self, api_key: str | None = None,
                 transport: Callable[[str, str, dict | None, str], tuple[int, dict]] = _http):
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        if not self.api_key:
            raise VenueError(f"no {API_KEY_ENV} in the environment; create one at "
                             f"runpod.io/console/user/settings")
        self._transport = transport

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        status, data = self._transport(method, path, body, self.api_key)
        if status >= 400:
            raise VenueError(f"runpod {method} {path} -> {status}: {data.get('error', data)}")
        return data

    def create(self, spec: PodSpec) -> Pod:
        d = self._call("POST", "/pods", spec.as_body())
        return Pod(id=d["id"], cost_per_hr=_cost(d.get("costPerHr")))

    def get(self, pod_id: str) -> Pod:
        d = self._call("GET", f"/pods/{pod_id}")
        ssh_port = None
        for m in d.get("portMappings") or []:
            # portMappings is documented as a list of maps; RunPod has also returned it as a
            # single {"22": 40123} dict, so accept both rather than crash on a live shape.
            if isinstance(m, dict):
                got = m.get("22") or (m.get("publicPort") if m.get("privatePort") == 22 else None)
                if got:
                    ssh_port = int(got)
        if isinstance(d.get("portMappings"), dict):
            ssh_port = ssh_port or int(d["portMappings"].get("22") or 0) or None
        return Pod(id=d["id"], host=d.get("publicIp") or None, port=ssh_port,
                   cost_per_hr=_cost(d.get("costPerHr")))

    def terminate(self, pod_id: str) -> None:
        self._call("DELETE", f"/pods/{pod_id}")

    def list(self, log: Callable[[str], None] = print) -> list[Pod]:
        """Every pod on the account. The reaper needs names and start times, nothing else.

        One malformed row must not abort the sweep — the other leaked pods would keep billing —
        so a row without an id is logged and skipped, never raised. An unexpected response
        shape is logged loudly rather than read as "no pods".
        """
        d = self._call("GET", "/pods")
        if isinstance(d, list):
            rows = d
        elif isinstance(d, dict) and isinstance(d.get("pods"), list):
            rows = d["pods"]
        else:
            log(f"[venue] WARNING GET /pods returned an unexpected shape, treating as empty: "
                f"{str(d)[:200]} — check runpod.io/console/pods by hand")
            rows = []
        pods: list[Pod] = []
        for p in rows:
            if not isinstance(p, dict) or not p.get("id"):
                log(f"[venue] WARNING skipping malformed pod record: {str(p)[:200]}")
                continue
            started = p.get("lastStartedAt")
            pods.append(Pod(id=str(p["id"]), name=str(p.get("name") or ""),
                            last_started_at=str(started) if started else None,
                            cost_per_hr=_cost(p.get("costPerHr"))))
        return pods


@dataclass
class PodSpec:
    """What to rent. Defaults mirror modal_app.py so a panel measured here is comparable."""

    name: str = "inference-server-instrument"
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    gpu_type_ids: list[str] = field(default_factory=lambda: ["NVIDIA A100 80GB PCIe"])
    gpu_count: int = 1
    # E4B weights are ~15GB and torch+cuda is another ~8GB, so the 50GB default is not slack.
    container_disk_gb: int = 80
    volume_gb: int = 100
    volume_mount_path: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)

    def as_body(self) -> dict[str, Any]:
        return {
            "name": self.name, "imageName": self.image, "computeType": "GPU",
            "gpuTypeIds": self.gpu_type_ids, "gpuCount": self.gpu_count,
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": self.volume_gb, "volumeMountPath": self.volume_mount_path,
            "ports": ["22/tcp"],
            "env": dict(self.env),
        }


# ------------------------------------------------------------------ shell out to the pod

# rsync excludes: ship the engine and the instruments, not 40GB of local runs and weights.
_SYNC_INCLUDE = ("src", "scripts", "pyproject.toml")

Shell = Callable[[list[str]], subprocess.CompletedProcess]


def _shell(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _ssh_base(pod: Pod) -> list[str]:
    # A rented box has a fresh host key every time, so strict checking would prompt and hang.
    # This is a throwaway machine we created seconds ago, not a server whose identity we know.
    return ["ssh", "-p", str(pod.port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            f"root@{pod.host}"]


def wait_until_ready(client: RunPodClient, pod_id: str, timeout_s: float = 300.0,
                     poll_s: float = 5.0, sleep: Callable[[float], None] = time.sleep) -> Pod:
    """Block until the pod has a public IP and an SSH port, or give up.

    Giving up still costs money — the pod exists — so the caller must terminate on this path.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pod = client.get(pod_id)
        if pod.reachable:
            return pod
        sleep(poll_s)
    raise VenueError(f"pod {pod_id} had no ssh endpoint after {timeout_s:.0f}s")


def sync_repo(pod: Pod, repo: str, shell: Shell = _shell) -> None:
    """Push the working tree. rsync, not git clone: the whole point is to measure THIS tree,
    including changes that are not pushed anywhere yet."""
    ssh = " ".join(_ssh_base(pod)[:-1])
    cmd = ["rsync", "-az", "--delete", "-e", ssh]
    cmd += [f"{repo.rstrip('/')}/{p}" for p in _SYNC_INCLUDE]
    cmd += [f"root@{pod.host}:/workspace/repo/"]
    r = shell(cmd)
    if r.returncode != 0:
        raise VenueError(f"rsync to pod failed: {r.stderr.strip()[:400]}")


def provision(pod: Pod, shell: Shell = _shell, timeout_s: int = 900) -> None:
    """Install the repo's dependencies on the pod.

    The base image carries torch and CUDA; fastapi, transformers, accelerate, httpx and
    prometheus-client are not there, so an instrument dies on its first import without this.
    `torch>=2.4` is already satisfied by the image, so pip skips the 2 GB download.
    """
    remote = (f"cd /workspace/repo && timeout {timeout_s} "
              f"pip install --quiet --no-input -e . 2>&1 | tail -20")
    r = shell(_ssh_base(pod) + [remote])
    if r.returncode != 0:
        raise VenueError(f"pip install failed on pod {pod.id}: "
                         f"{(r.stderr or r.stdout).strip()[-600:]}")


def run_remote(pod: Pod, script: str, env: dict[str, str], shell: Shell = _shell,
               timeout_s: int = 3600) -> str:
    """Run one instrument and return its stdout.

    Provenance env is exported remotely for the same reason run_instrument.sh exports it locally:
    the box has no git repo, so a panel measured here cannot name its own sha.

    A non-zero exit always raises `InstrumentFailed`, which carries the stdout. Deciding whether
    that exit was the instrument speaking (a `gate` verdict, an honest `{"error": ...}` report)
    or a broken run is `run_instrument`'s job, so the policy lives in exactly one place.
    """
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
    remote = (f"cd /workspace/repo && export PYTHONPATH=/workspace/repo/src && "
              f"{exports} timeout {timeout_s} python {script}")
    r = shell(_ssh_base(pod) + [remote])
    if r.returncode != 0:
        raise InstrumentFailed(f"instrument exited {r.returncode} on pod {pod.id}: "
                               f"{(r.stderr or r.stdout).strip()[-600:]}", stdout=r.stdout or "")
    return r.stdout


def run_instrument(script: str, env: dict[str, str], *, repo: str,
                   spec: PodSpec | None = None, client: RunPodClient | None = None,
                   shell: Shell = _shell, log: Callable[[str], None] = print,
                   ready_timeout_s: float = 300.0, run_timeout_s: int = 3600,
                   sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Rent -> sync -> run -> parse -> terminate. The whole venue, one call.

    Terminate is in a `finally` and swallows its own errors: a failure to tear down must not mask
    the real exception, but it must also never be skipped. Nothing else in this repo can leave a
    billed resource running.

    `run_timeout_s` is the instrument's own wall budget, and it was previously unreachable: a run
    that needs longer than the default hour is killed mid-print, the payload never closes, and the
    whole rental yields nothing. An instrument that loads a model AND compiles a decode ladder is
    already within a factor of two of it (knowledge/kb-20260902-008.json: ~21 minutes of compile
    at MAX_BATCH_SIZE=256), so it has to be the caller's choice.
    """
    client = client or RunPodClient()
    spec = spec or PodSpec()
    spec.env = {**spec.env, **env}

    pod = client.create(spec)
    log(f"[venue] pod {pod.id} created (${pod.cost_per_hr:.2f}/hr)")
    try:
        pod = wait_until_ready(client, pod.id, timeout_s=ready_timeout_s, sleep=sleep)
        log(f"[venue] ssh up at {pod.host}:{pod.port}")
        sync_repo(pod, repo, shell=shell)
        log("[venue] installing dependencies")
        provision(pod, shell=shell)
        try:
            out = run_remote(pod, script, env, shell=shell, timeout_s=run_timeout_s)
        except InstrumentFailed as e:
            # A non-zero exit is the instrument SPEAKING when its payload is a verdict (`gate`)
            # or an honest failure report (`error`) — a gate exits by its verdict so a direct run
            # is usable alone, and a replay that lost one config reports the rest plus `error`.
            # Keeping that payload saves GPU-minutes already billed. Anything else — panels only,
            # or no payload at all — is a broken run and still raises, which is the rule that
            # stops a crash from reading as "no effect measured".
            if not reports_its_own_failure(e.stdout):
                raise
            log(f"[venue] instrument exited non-zero and reported why — keeping its payload: {e}")
            return extract_payload(e.stdout)
        return extract_payload(out)
    finally:
        try:
            client.terminate(pod.id)
            log(f"[venue] pod {pod.id} terminated")
        except Exception as e:                      # noqa: BLE001 — see docstring
            log(f"[venue] WARNING could not terminate {pod.id}: {e} — "
                f"check runpod.io/console/pods, it is still billing")


# ------------------------------------------------------------------ the reaper

def _parse_ts(s: str) -> datetime | None:
    """ISO-8601 to an aware datetime; None when the string is not one."""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def reap_pods(client: RunPodClient, name: str = PodSpec.name, since: str | None = None,
              log: Callable[[str], None] = print) -> list[str]:
    """Terminate every pod carrying `name`; with `since` (ISO-8601), only those started at or
    after it. Returns the ids terminated.

    The CI cleanup step. A runner killed by its timeout never reaches `run_instrument`'s
    finally, and that pod bills until someone finds it in a web console. The name is the proof
    of ownership — only pods this repo named are candidates; `since` is the guard against a
    concurrent local run that used the same name. A pod whose start time is missing or
    unreadable is still ours by name, so it is terminated, loudly: "cannot tell how old" must
    not mean "keeps billing".
    """
    cutoff = _parse_ts(since) if since else None
    if since and cutoff is None:
        raise VenueError(f"--since {since!r} is not an ISO-8601 timestamp")
    reaped: list[str] = []
    for pod in client.list(log=log):
        if pod.name != name:
            continue
        started = _parse_ts(pod.last_started_at) if pod.last_started_at else None
        if cutoff and started and started < cutoff:
            log(f"[reap] leaving {pod.id}: started {pod.last_started_at}, before {since}")
            continue
        if cutoff and started is None:
            log(f"[reap] {pod.id} has no readable start time ({pod.last_started_at!r}); "
                f"it carries our name, so it goes")
        client.terminate(pod.id)
        log(f"[reap] terminated {pod.id} ({name}, ${pod.cost_per_hr:.2f}/hr)")
        reaped.append(pod.id)
    return reaped
