"""lab.vm against a fake `crusoe` CLI, ssh and rsync on PATH: the lifecycle it drives, not the cloud."""

from __future__ import annotations

import json
import os
import stat

import pytest

from lab import vm as labvm
from lab.providers import ProviderError, crusoe

FAKE_CRUSOE = r'''#!/usr/bin/env python3
"""Fake crusoe CLI: state lives in $FAKE_STATE, every call is appended to $FAKE_LOG."""
import json, os, sys
state_path, log = os.environ["FAKE_STATE"], os.environ["FAKE_LOG"]
args = sys.argv[1:]
open(log, "a").write(json.dumps(args) + "\n")
db = json.load(open(state_path)) if os.path.exists(state_path) else {}
def save(): json.dump(db, open(state_path, "w"))
if args[:3] == ["compute", "vms", "list"]:
    print(json.dumps([{"name": n, "state": v["state"], "type": "a100-80gb.1x",
                       "network_interfaces": [{"ips": [{"public_ipv4": {"address": "10.0.0.7"}}]}]}
                      for n, v in db.items()]))
elif args[:3] == ["compute", "vms", "create"]:
    db[args[args.index("--name") + 1]] = {"state": "STATE_RUNNING"}; save()
elif args[:3] == ["compute", "vms", "start"]:
    db[args[3]]["state"] = "RUNNING"; save()
elif args[:3] == ["compute", "vms", "stop"]:
    db[args[3]]["state"] = "STOPPED"; save()
elif args[:3] == ["compute", "vms", "types"]:
    print("a100-80gb.1x\nl40s-48gb.1x")
else:
    sys.stderr.write("unknown\n"); sys.exit(2)
'''

FAKE_TOOL = r'''#!/usr/bin/env python3
import json, os, sys
open(os.environ["FAKE_LOG"], "a").write(json.dumps([os.path.basename(sys.argv[0])] + sys.argv[1:]) + "\n")
# FAKE_EXIT applies to the user's command only (the one that cd's into the repo), not to ssh probes
sys.exit(int(os.environ.get("FAKE_EXIT", "0")) if any(a.startswith("cd ") for a in sys.argv) else 0)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("crusoe", FAKE_CRUSOE), ("ssh", FAKE_TOOL), ("rsync", FAKE_TOOL)):
        p = bin_dir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("LAB_VM_PROVIDER", "crusoe")
    monkeypatch.setenv("LAB_VM_KEYFILE", str(tmp_path / "key.pub"))
    monkeypatch.setattr(labvm, "KNOWN_HOSTS_DIR", tmp_path / "kh")
    monkeypatch.setattr(labvm, "POLL_S", 0.0)
    monkeypatch.setattr(labvm, "SSH_TIMEOUT_S", 0.05)
    return tmp_path


def calls(tmp_path):
    log = tmp_path / "calls.log"
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def test_start_creates_an_absent_vm(fake):
    vm = labvm.VM(name="t1")
    record = labvm.start(vm)
    assert record.state == "running" and record.ip == "10.0.0.7"
    create = next(c for c in calls(fake) if c[:3] == ["compute", "vms", "create"])
    assert create[create.index("--type") + 1] == vm.provider.type


def test_start_starts_a_stopped_vm_and_stop_waits(fake):
    vm = labvm.VM(name="t2")
    labvm.start(vm)
    labvm.stop(vm)
    assert vm.provider.get("t2").state == "stopped"
    labvm.start(vm)
    kinds = [c[2] for c in calls(fake) if c[:2] == ["compute", "vms"]]
    assert kinds.count("create") == 1 and kinds.count("start") == 1 and kinds.count("stop") == 1


def test_stop_is_a_no_op_when_absent_or_stopped(fake):
    labvm.stop(labvm.VM(name="t3"))
    vm = labvm.VM(name="t3b")
    labvm.start(vm)
    labvm.stop(vm)
    labvm.stop(vm)
    stops = [c for c in calls(fake) if c[:3] == ["compute", "vms", "stop"]]
    assert stops == [["compute", "vms", "stop", "t3b"]]


def test_crusoe_state_accepts_both_spellings(fake):
    vm = labvm.VM(name="t3c")
    labvm.start(vm)                                    # the fake writes STATE_RUNNING on create
    assert vm.provider.get("t3c").state == "running"
    vm.provider.stop("t3c")                            # and STOPPED on stop
    assert vm.provider.get("t3c").state == "stopped"


def test_run_pushes_runs_in_repo_dir_and_fetches(fake):
    vm = labvm.VM(name="t4", user="u", remote_dir="~/repo")
    local = fake / "out"
    assert labvm.run(vm, "python -m lab.profile", fetch_dir="lab/runs", local=local) == 0
    log = calls(fake)
    rsyncs = [c for c in log if c[0] == "rsync"]
    assert rsyncs[0][-1] == "u@10.0.0.7:~/repo/" and "--filter=:- .gitignore" in rsyncs[0]
    assert rsyncs[1][-2] == "u@10.0.0.7:~/repo/lab/runs/"
    sshs = [c for c in log if c[0] == "ssh"]
    cmd = sshs[-1][-1]
    assert cmd.startswith("cd ~/repo && LAB_GIT_SHA=") and cmd.endswith('PATH="$PWD/.venv/bin:$PATH" python -m lab.profile')
    assert any(a.startswith("UserKnownHostsFile=") and a.endswith("t4.known_hosts") for a in sshs[-1])


def test_run_returns_the_remote_exit_code_and_still_fetches(fake, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    assert labvm.run(labvm.VM(name="t5"), "false", fetch_dir="lab/runs", local=fake / "out") == 3
    assert any(c[0] == "rsync" and c[-2].endswith("lab/runs/") for c in calls(fake))


def test_login_defaults_to_the_provider(fake, monkeypatch):
    assert labvm.VM().login == "ubuntu"
    monkeypatch.setenv("LAB_VM_USER", "me")
    assert labvm.VM().login == "me"


def test_provider_reads_env_at_construction(fake, monkeypatch):
    monkeypatch.setenv("LAB_VM_TYPE", "l40s-48gb.1x")
    assert crusoe.Crusoe().type == "l40s-48gb.1x"


def test_cli_status_types_and_separator(fake, capsys, monkeypatch):
    assert labvm.main(["status"]) == 0
    assert "absent" in capsys.readouterr().out
    assert labvm.main(["types"]) == 0
    assert "a100-80gb.1x" in capsys.readouterr().out
    seen = {}
    monkeypatch.setattr(labvm, "run", lambda vm, command, **kw: seen.setdefault("cmd", command) and 0)
    labvm.main(["run", "--", "pytest", "--", "tests"])
    assert seen["cmd"] == "pytest -- tests"


def test_missing_cli_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ProviderError, match="not installed"):
        crusoe.Crusoe().get("t7")


def test_unknown_provider(monkeypatch):
    monkeypatch.setenv("LAB_VM_PROVIDER", "nimbus")
    with pytest.raises(ProviderError, match="unknown provider"):
        labvm.VM()


def test_ssh_uses_the_private_half_of_the_keyfile(fake, monkeypatch):
    monkeypatch.setenv("LAB_VM_KEYFILE", "~/.ssh/lab_ed25519.pub")
    opts = labvm.VM().ssh_opts
    assert opts[:2] == ["-i", os.path.expanduser("~/.ssh/lab_ed25519")]


def test_rsync_quotes_a_keyfile_path_with_spaces(fake, monkeypatch):
    import shlex
    monkeypatch.setenv("LAB_VM_KEYFILE", "/Users/A B/.ssh/lab key.pub")
    labvm.run(labvm.VM(name="t5"), "true")
    e = next(c for c in calls(fake) if c[0] == "rsync")[3]
    assert shlex.split(e)[1:3] == ["-i", "/Users/A B/.ssh/lab key"]
