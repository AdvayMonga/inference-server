"""The GPU arm: one persistent Crusoe VM driven through the `crusoe` CLI (usage: lab/README.md)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KNOWN_HOSTS_DIR = Path.home() / ".cache" / "inference-server-lab"
POLL_S = 5.0             # between state and ssh probes
START_TIMEOUT_S = 600.0
STOP_TIMEOUT_S = 300.0
SSH_TIMEOUT_S = 300.0


class CrusoeError(RuntimeError):
    """The CLI, ssh or rsync refused; stderr is in the message."""


def _env(name: str, default: str):
    return field(default_factory=lambda: os.environ.get(name, default))


@dataclass(frozen=True)
class VM:
    """Read from env at construction, so the same code serves any account or GPU."""
    name: str = _env("LAB_VM", "lab-gpu")
    type: str = _env("LAB_VM_TYPE", "a100-80gb.1x")
    location: str = _env("LAB_VM_LOCATION", "us-east1-a")
    image: str = _env("LAB_VM_IMAGE", "ubuntu22.04-nvidia-slurm:latest")
    user: str = _env("LAB_VM_USER", "ubuntu")
    keyfile: str = _env("LAB_VM_KEYFILE", "~/.ssh/id_ed25519.pub")
    remote_dir: str = _env("LAB_VM_DIR", "~/inference-server")

    @property
    def known_hosts(self) -> Path:
        """Per-VM host keys: a recreated VM may get a reused IP, and a stale global entry would block every probe."""
        return KNOWN_HOSTS_DIR / f"{self.name}.known_hosts"

    @property
    def ssh_opts(self) -> list[str]:
        return ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10", "-o", f"UserKnownHostsFile={self.known_hosts}"]


def _cli(*args: str) -> str:
    try:
        out = subprocess.run(["crusoe", *args], capture_output=True, text=True)
    except FileNotFoundError:
        raise CrusoeError("`crusoe` CLI not installed: https://docs.crusoecloud.com/quickstart/installing-the-cli/")
    if out.returncode != 0:
        raise CrusoeError(f"crusoe {' '.join(args)}: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def get(vm: VM) -> dict | None:
    """The VM's record from the project listing, or None when no VM has that name."""
    listing = json.loads(_cli("compute", "vms", "list", "--json"))
    if isinstance(listing, dict):                      # some CLI versions wrap the array
        listing = listing.get("items") or listing.get("vms") or []
    return next((r for r in listing if r.get("name") == vm.name), None)


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
    vm.known_hosts.unlink(missing_ok=True)
    _cli("compute", "vms", "create", "--name", vm.name, "--type", vm.type, "--location", vm.location,
         "--image", vm.image, "--keyfile", os.path.expanduser(vm.keyfile))


def _wait_state(vm: VM, wanted: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        record = get(vm)
        if record is not None and state(record) == wanted:
            return record
        if time.monotonic() > deadline:
            raise CrusoeError(f"{vm.name} not {wanted} after {timeout_s:.0f}s "
                              f"(state: {state(record) if record else 'absent'})")
        time.sleep(POLL_S)


def start(vm: VM) -> dict:
    record = get(vm)
    if record is None:
        create(vm)
    elif state(record) == "running":
        return record
    elif state(record) == "stopped":
        _cli("compute", "vms", "start", vm.name)
    return _wait_state(vm, "running", START_TIMEOUT_S)


def stop(vm: VM) -> None:
    """Stopped on-demand VMs keep their disk and bill nothing; this is the only way a session ends."""
    record = get(vm)
    if record is None or state(record) == "stopped":
        return
    _cli("compute", "vms", "stop", vm.name)
    _wait_state(vm, "stopped", STOP_TIMEOUT_S)


def ssh_target(vm: VM, record: dict) -> str:
    return f"{vm.user}@{ip(record)}"


def ssh(vm: VM, record: dict, command: str, *, check: bool = True, capture: bool = False,
        timeout_s: float | None = None) -> subprocess.CompletedProcess:
    vm.known_hosts.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["ssh", *vm.ssh_opts, ssh_target(vm, record), command], text=True,
                         capture_output=capture, timeout=timeout_s)
    if check and out.returncode != 0:
        raise CrusoeError(f"ssh {command!r} exited {out.returncode}"
                          + (f": {out.stderr.strip()}" if capture else ""))
    return out


def wait_ssh(vm: VM, record: dict) -> None:
    deadline = time.monotonic() + SSH_TIMEOUT_S
    last = ""
    while True:
        try:
            out = ssh(vm, record, "true", check=False, capture=True, timeout_s=15)
            if out.returncode == 0:
                return
            last = out.stderr.strip().splitlines()[-1] if out.stderr.strip() else f"exit {out.returncode}"
        except subprocess.TimeoutExpired:
            last = "connect timeout"
        if time.monotonic() > deadline:
            raise CrusoeError(f"no ssh to {ssh_target(vm, record)} after {SSH_TIMEOUT_S:.0f}s: {last}")
        time.sleep(POLL_S)


def _rsync(vm: VM, *args: str) -> None:
    out = subprocess.run(["rsync", "-az", "-e", f"ssh {' '.join(vm.ssh_opts)}", *args],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise CrusoeError(f"rsync {args[-2]} -> {args[-1]}: {out.stderr.strip()}")


def push(vm: VM, record: dict) -> None:
    """The working tree, minus .git and everything .gitignore excludes (secrets, weights, venvs, runs)."""
    _rsync(vm, "--delete", "--exclude=.git", "--filter=:- .gitignore",
           f"{REPO}/", f"{ssh_target(vm, record)}:{vm.remote_dir}/")


def fetch(vm: VM, record: dict, remote: str, local: Path) -> None:
    """`remote` is relative to the repo dir on the VM."""
    local.mkdir(parents=True, exist_ok=True)
    _rsync(vm, f"{ssh_target(vm, record)}:{vm.remote_dir}/{remote}/", f"{local}/")


def run(vm: VM, command: str, *, fetch_dir: str | None = None, local: Path | None = None,
        push_tree: bool = True) -> int:
    """Start the VM if needed, push the tree, run `command` in the repo dir with its venv on PATH, fetch `fetch_dir`. Leaves the VM running."""
    record = start(vm)
    wait_ssh(vm, record)
    if push_tree:
        push(vm, record)
    t0 = time.monotonic()
    out = ssh(vm, record, f'cd {vm.remote_dir} && PATH="$PWD/.venv/bin:$PATH" {command}', check=False)
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
    r.add_argument("--fetch", help="remote dir, relative to the repo, to bring back")
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
    words = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not words:
        ap.error("run needs a command")
    return run(vm, " ".join(shlex.quote(w) for w in words), fetch_dir=args.fetch,
               local=Path(args.local) if args.local else None, push_tree=not args.no_push)


if __name__ == "__main__":
    raise SystemExit(main())
