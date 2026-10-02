"""The GPU arm: one persistent Crusoe VM driven through the `crusoe` CLI (usage: lab/README.md)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLL_S = 5.0             # between state and ssh probes
START_TIMEOUT_S = 600.0
STOP_TIMEOUT_S = 300.0
SSH_TIMEOUT_S = 300.0
RSYNC_EXCLUDE = (".git", "venv", ".venv", "__pycache__", "lab/runs", "runs", "sandbox", "*.pt")


class CrusoeError(RuntimeError):
    """The CLI, ssh or rsync refused; stderr is in the message."""


@dataclass(frozen=True)
class VM:
    """Everything about the VM comes from env, so the same code serves any account or GPU."""
    name: str = os.environ.get("LAB_VM", "lab-gpu")
    type: str = os.environ.get("LAB_VM_TYPE", "a100-80gb.1x")
    location: str = os.environ.get("LAB_VM_LOCATION", "us-east1-a")
    image: str = os.environ.get("LAB_VM_IMAGE", "ubuntu22.04-nvidia-slurm:latest")
    user: str = os.environ.get("LAB_VM_USER", "ubuntu")
    keyfile: str = os.environ.get("LAB_VM_KEYFILE", "~/.ssh/id_ed25519.pub")
    remote_dir: str = os.environ.get("LAB_VM_DIR", "~/inference-server")


def _cli(*args: str) -> str:
    try:
        out = subprocess.run(["crusoe", *args], capture_output=True, text=True)
    except FileNotFoundError:
        raise CrusoeError("`crusoe` CLI not installed: https://docs.crusoecloud.com/quickstart/installing-the-cli/")
    if out.returncode != 0:
        raise CrusoeError(f"crusoe {' '.join(args)}: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def get(vm: VM) -> dict | None:
    """The VM's record, or None when it does not exist."""
    try:
        return json.loads(_cli("compute", "vms", "get", vm.name, "--json"))
    except CrusoeError as e:
        if "not found" in str(e).lower():
            return None
        raise


def state(record: dict) -> str:
    """'running', 'stopped', ... whichever spelling the API uses (RUNNING, STATE_RUNNING)."""
    return str(record.get("state", "")).lower().removeprefix("state_")


def ip(record: dict) -> str:
    for nic in record.get("network_interfaces", []):
        for addr in nic.get("ips", []):
            public = (addr.get("public_ipv4") or {}).get("address")
            if public:
                return public
    raise CrusoeError(f"{record.get('name')} has no public IPv4 yet")


def create(vm: VM) -> None:
    _cli("compute", "vms", "create", "--name", vm.name, "--type", vm.type, "--location", vm.location,
         "--image", vm.image, "--keyfile", os.path.expanduser(vm.keyfile))


def _wait_state(vm: VM, wanted: str, timeout_s: float, poll_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        record = get(vm)
        if record is not None and state(record) == wanted:
            return record
        if time.monotonic() > deadline:
            raise CrusoeError(f"{vm.name} not {wanted} after {timeout_s:.0f}s "
                              f"(state: {state(record) if record else 'absent'})")
        time.sleep(poll_s)


def start(vm: VM) -> dict:
    record = get(vm)
    if record is None:
        create(vm)
    elif state(record) == "running":
        return record
    elif state(record) == "stopped":
        _cli("compute", "vms", "start", vm.name)
    return _wait_state(vm, "running", START_TIMEOUT_S, POLL_S)


def stop(vm: VM) -> None:
    """Stopped on-demand VMs keep their disk and bill nothing; this is the only way a session ends."""
    record = get(vm)
    if record is None or state(record) == "stopped":
        return
    _cli("compute", "vms", "stop", vm.name)
    _wait_state(vm, "stopped", STOP_TIMEOUT_S, POLL_S)


def ssh_target(vm: VM, record: dict) -> str:
    return f"{vm.user}@{ip(record)}"


SSH_OPTS = ("-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")


def ssh(vm: VM, record: dict, command: str, *, check: bool = True, capture: bool = False,
        timeout_s: float | None = None) -> subprocess.CompletedProcess:
    out = subprocess.run(["ssh", *SSH_OPTS, ssh_target(vm, record), command], text=True,
                         capture_output=capture, timeout=timeout_s)
    if check and out.returncode != 0:
        raise CrusoeError(f"ssh {command!r} exited {out.returncode}"
                          + (f": {out.stderr.strip()}" if capture else ""))
    return out


def wait_ssh(vm: VM, record: dict) -> None:
    deadline = time.monotonic() + SSH_TIMEOUT_S
    while True:
        try:
            if ssh(vm, record, "true", check=False, capture=True, timeout_s=15).returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() > deadline:
            raise CrusoeError(f"no ssh to {ssh_target(vm, record)} after {SSH_TIMEOUT_S:.0f}s")
        time.sleep(POLL_S)


def push(vm: VM, record: dict) -> None:
    """The working tree, not a commit: what is measured is what is on disk here."""
    excludes = [f"--exclude={e}" for e in RSYNC_EXCLUDE]
    out = subprocess.run(["rsync", "-az", "--delete", *excludes, "-e", f"ssh {' '.join(SSH_OPTS)}",
                          f"{REPO}/", f"{ssh_target(vm, record)}:{vm.remote_dir}/"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise CrusoeError(f"rsync push: {out.stderr.strip()}")


def fetch(vm: VM, record: dict, remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["rsync", "-az", "-e", f"ssh {' '.join(SSH_OPTS)}",
                          f"{ssh_target(vm, record)}:{remote}/", f"{local}/"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise CrusoeError(f"rsync fetch {remote}: {out.stderr.strip()}")


def run(vm: VM, command: str, *, fetch_dir: str | None = None, local: Path | None = None,
        push_tree: bool = True) -> int:
    """Start the VM if needed, push the tree, run `command` in the repo dir, fetch `fetch_dir`. Leaves the VM running."""
    record = start(vm)
    wait_ssh(vm, record)
    if push_tree:
        push(vm, record)
    t0 = time.monotonic()
    out = ssh(vm, record, f"cd {vm.remote_dir} && {command}", check=False)
    print(f"[crusoe] exit {out.returncode} after {time.monotonic() - t0:.0f}s on {vm.name}",
          file=sys.stderr)
    if fetch_dir:
        fetch(vm, record, fetch_dir, local or REPO / "lab" / "runs")
    return out.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("(")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="state, type and ip")
    sub.add_parser("start", help="create or start, wait for ssh")
    sub.add_parser("stop", help="stop; disk kept, billing ends")
    sub.add_parser("types", help="VM types the account can rent")
    sub.add_parser("setup", help="run lab/vm-setup.sh on the VM")
    r = sub.add_parser("run", help="push the tree, run a command in it, fetch an output dir")
    r.add_argument("--fetch", help="remote dir (relative to the repo) to bring back")
    r.add_argument("--local", help="where to put it (default lab/runs)")
    r.add_argument("--no-push", action="store_true")
    r.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    vm = VM()

    if args.cmd == "status":
        record = get(vm)
        if record is None:
            print(f"{vm.name}: absent (would create {vm.type} in {vm.location})")
        else:
            s = state(record)
            print(f"{vm.name}: {s} {record.get('type', '')} "
                  + (ip(record) if s == "running" else ""))
        return 0
    if args.cmd == "start":
        record = start(vm)
        wait_ssh(vm, record)
        print(ssh_target(vm, record))
        return 0
    if args.cmd == "stop":
        stop(vm)
        print(f"{vm.name}: stopped")
        return 0
    if args.cmd == "types":
        print(_cli("compute", "vms", "types"), end="")
        return 0
    if args.cmd == "setup":
        return run(vm, "bash lab/vm-setup.sh")
    command = " ".join(shlex.quote(c) for c in args.command if c != "--")
    if not command:
        ap.error("run needs a command")
    return run(vm, command, fetch_dir=args.fetch, local=Path(args.local) if args.local else None,
               push_tree=not args.no_push)


if __name__ == "__main__":
    raise SystemExit(main())
