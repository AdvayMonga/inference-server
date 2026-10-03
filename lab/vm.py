"""The GPU arm: one persistent VM on a cloud of your choice, same lifecycle everywhere (usage: lab/README.md)."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from lab import providers
from lab.providers import ProviderError, Record, env_field

REPO = Path(__file__).resolve().parents[1]
KNOWN_HOSTS_DIR = Path.home() / ".cache" / "inference-server-lab"
POLL_S = 5.0             # between state and ssh probes
START_TIMEOUT_S = 900.0
STOP_TIMEOUT_S = 600.0
SSH_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class VM:
    """Read from env at construction. The provider supplies GPU type, location, image and the login user."""
    name: str = env_field("LAB_VM", "lab-gpu")
    keyfile: str = env_field("LAB_VM_KEYFILE", "~/.ssh/id_ed25519.pub")
    remote_dir: str = env_field("LAB_VM_DIR", "~/inference-server")
    user: str = env_field("LAB_VM_USER", "")
    provider: object = field(default_factory=lambda: providers.load(os.environ.get("LAB_VM_PROVIDER", "verda")))

    @property
    def login(self) -> str:
        return self.user or self.provider.default_user

    @property
    def known_hosts(self) -> Path:
        """Per-VM host keys: a recreated VM may get a reused IP, and a stale global entry would block every probe."""
        return KNOWN_HOSTS_DIR / f"{self.name}.known_hosts"

    @property
    def ssh_opts(self) -> list[str]:
        identity = os.path.expanduser(self.keyfile.removesuffix(".pub"))   # the private half of the uploaded key
        return ["-i", identity, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10", "-o", f"UserKnownHostsFile={self.known_hosts}"]


def _wait(vm: VM, wanted: str, timeout_s: float) -> Record:
    """Until the provider reports `wanted`; `running` also needs a public ip, which can lag the state."""
    deadline = time.monotonic() + timeout_s
    while True:
        record = vm.provider.get(vm.name)
        if record is not None and record.state == wanted and (wanted != "running" or record.ip):
            return record
        if record is not None and record.state in ("error", "no_capacity"):
            raise ProviderError(f"{vm.name}: {record.detail}")
        if time.monotonic() > deadline:
            raise ProviderError(f"{vm.name} not {wanted} after {timeout_s:.0f}s "
                                f"(state: {record.detail if record else 'absent'})")
        time.sleep(POLL_S)


def start(vm: VM) -> Record:
    record = vm.provider.get(vm.name)
    if record is None:
        vm.known_hosts.unlink(missing_ok=True)
        vm.provider.create(vm.name, vm.keyfile)
    elif record.state == "running" and record.ip:
        return record
    elif record.state in ("stopped", "offline"):
        vm.provider.start(vm.name)
    return _wait(vm, "running", START_TIMEOUT_S)


def stop(vm: VM) -> None:
    """Ends GPU billing (Crusoe: stop; Verda: hibernate). The disk survives; this is how a session ends."""
    record = vm.provider.get(vm.name)
    if record is None or record.state == "stopped":
        return
    vm.provider.stop(vm.name)
    _wait(vm, "stopped", STOP_TIMEOUT_S)


def ssh_target(vm: VM, record: Record) -> str:
    if not record.ip:
        raise ProviderError(f"{vm.name} has no public ip yet ({record.detail})")
    return f"{vm.login}@{record.ip}"


def ssh(vm: VM, record: Record, command: str, *, check: bool = True, capture: bool = False,
        timeout_s: float | None = None) -> subprocess.CompletedProcess:
    vm.known_hosts.parent.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["ssh", *vm.ssh_opts, ssh_target(vm, record), command], text=True,
                         capture_output=capture, timeout=timeout_s)
    if check and out.returncode != 0:
        raise ProviderError(f"ssh {command!r} exited {out.returncode}"
                            + (f": {out.stderr.strip()}" if capture else ""))
    return out


def wait_ssh(vm: VM, record: Record) -> None:
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
            raise ProviderError(f"no ssh to {ssh_target(vm, record)} after {SSH_TIMEOUT_S:.0f}s: {last}")
        time.sleep(POLL_S)


def _rsync(vm: VM, *args: str) -> None:
    out = subprocess.run(["rsync", "-az", "-e", f"ssh {shlex.join(vm.ssh_opts)}", *args],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise ProviderError(f"rsync {args[-2]} -> {args[-1]}: {out.stderr.strip()}")


def push(vm: VM, record: Record) -> None:
    """The working tree, minus .git and everything .gitignore excludes (secrets, weights, venvs, runs)."""
    # .venv is named explicitly: a gitignore-only exclusion did not protect it from --delete on the VM.
    _rsync(vm, "--delete", "--exclude=.git", "--exclude=.venv", "--exclude=__pycache__", "--filter=:- .gitignore",
           f"{REPO}/", f"{ssh_target(vm, record)}:{vm.remote_dir}/")


def fetch(vm: VM, record: Record, remote: str, local: Path) -> None:
    """`remote` is relative to the repo dir on the VM."""
    local.mkdir(parents=True, exist_ok=True)
    _rsync(vm, f"{ssh_target(vm, record)}:{vm.remote_dir}/{remote}/", f"{local}/")


def local_sha() -> str | None:
    """The working tree's HEAD, shipped to the VM as LAB_GIT_SHA because the pushed tree carries no .git."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() or None


def run(vm: VM, command: str, *, fetch_dir: str | None = None, local: Path | None = None,
        push_tree: bool = True) -> int:
    """Start the VM if needed, push the tree, run `command` in the repo dir with its venv on PATH, fetch `fetch_dir`. Leaves the VM running."""
    record = start(vm)
    wait_ssh(vm, record)
    if push_tree:
        push(vm, record)
    sha = local_sha()
    env = f"LAB_GIT_SHA={sha} " if sha else ""
    t0 = time.monotonic()
    out = ssh(vm, record, f'cd {vm.remote_dir} && {env}PATH="$PWD/.venv/bin:$PATH" {command}', check=False)
    print(f"[lab.vm] exit {out.returncode} after {time.monotonic() - t0:.0f}s on {vm.name}",
          file=sys.stderr)
    if fetch_dir:
        fetch(vm, record, fetch_dir, local or REPO / "lab" / "runs")
    return out.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("(")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="state, type and ip")
    sub.add_parser("start", help="create or start, wait for ssh")
    sub.add_parser("stop", help="end GPU billing; disk kept")
    sub.add_parser("types", help="what the account can rent right now")
    sub.add_parser("setup", help="run lab/vm-setup.sh on the VM")
    r = sub.add_parser("run", help="push the tree, run a command in it, fetch an output dir")
    r.add_argument("--fetch", help="remote dir, relative to the repo, to bring back")
    r.add_argument("--local", help="where to put it (default lab/runs)")
    r.add_argument("--no-push", action="store_true")
    r.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    vm = VM()
    p = vm.provider

    if args.cmd == "status":
        record = p.get(vm.name)
        if record is None:
            print(f"{vm.name}: absent (would create {p.type} in {p.location} on {type(p).__name__.lower()})")
        else:
            print(f"{vm.name}: {record.state} {record.type} {record.ip or ''} [{record.detail}]")
        return 0
    if args.cmd == "start":
        record = start(vm)
        wait_ssh(vm, record)
        print(ssh_target(vm, record))
        return 0
    if args.cmd == "stop":
        stop(vm)
        print(f"{vm.name}: {p.stop_word}d, GPU billing ended")
        return 0
    if args.cmd == "types":
        print(p.types(), end="")
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
