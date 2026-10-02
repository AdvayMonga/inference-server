"""lab.crusoe against a fake `crusoe` CLI, ssh and rsync on PATH: the lifecycle it drives, not the cloud."""

from __future__ import annotations

import json
import os
import stat

import pytest

from lab import crusoe

FAKE_CRUSOE = r'''#!/usr/bin/env python3
"""Fake crusoe CLI: state lives in $FAKE_STATE, every call is appended to $FAKE_LOG."""
import json, os, sys
state_path, log = os.environ["FAKE_STATE"], os.environ["FAKE_LOG"]
args = sys.argv[1:]
open(log, "a").write(json.dumps(args) + "\n")
db = json.load(open(state_path)) if os.path.exists(state_path) else {}
def save(): json.dump(db, open(state_path, "w"))
if args[:3] == ["compute", "vms", "get"]:
    name = args[3]
    if name not in db:
        sys.stderr.write(f"vm {name} not found\n"); sys.exit(1)
    print(json.dumps({"name": name, "state": db[name]["state"], "type": "a100-80gb.1x",
                      "network_interfaces": [{"ips": [{"public_ipv4": {"address": "10.0.0.7"}}]}]}))
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
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
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
    monkeypatch.setenv("LAB_VM_KEYFILE", str(tmp_path / "key.pub"))
    monkeypatch.setattr(crusoe, "POLL_S", 0.0)
    monkeypatch.setattr(crusoe, "SSH_TIMEOUT_S", 0.05)
    return tmp_path


def calls(tmp_path):
    log = tmp_path / "calls.log"
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def test_start_creates_an_absent_vm(fake):
    vm = crusoe.VM(name="t1")
    record = crusoe.start(vm)
    assert crusoe.state(record) == "running" and crusoe.ip(record) == "10.0.0.7"
    create = next(c for c in calls(fake) if c[:3] == ["compute", "vms", "create"])
    assert "--type" in create and create[create.index("--type") + 1] == vm.type


def test_start_starts_a_stopped_vm_and_stop_waits(fake):
    vm = crusoe.VM(name="t2")
    crusoe.start(vm)
    crusoe.stop(vm)
    assert crusoe.state(crusoe.get(vm)) == "stopped"
    crusoe.start(vm)
    kinds = [c[2] for c in calls(fake) if c[:2] == ["compute", "vms"]]
    assert kinds.count("create") == 1 and kinds.count("start") == 1 and kinds.count("stop") == 1


def test_stop_is_a_no_op_when_absent_or_stopped(fake):
    vm = crusoe.VM(name="t3")
    crusoe.stop(vm)
    assert all(c[2] != "stop" for c in calls(fake) if c[:2] == ["compute", "vms"])


def test_run_pushes_runs_in_repo_dir_and_fetches(fake):
    vm = crusoe.VM(name="t4", user="u", remote_dir="~/repo")
    local = fake / "out"
    rc = crusoe.run(vm, "python -m lab.profile", fetch_dir="lab/runs", local=local)
    assert rc == 0 and local.is_dir()
    log = calls(fake)
    rsyncs = [c for c in log if c[0] == "rsync"]
    assert rsyncs[0][-1] == "u@10.0.0.7:~/repo/" and any(a.startswith("--exclude=.git") for a in rsyncs[0])
    assert rsyncs[1][-2] == "u@10.0.0.7:lab/runs/"
    sshs = [c for c in log if c[0] == "ssh"]
    assert sshs[-1][-1] == "cd ~/repo && python -m lab.profile"


def test_run_reports_remote_exit_code(fake, monkeypatch):
    vm = crusoe.VM(name="t5")
    crusoe.start(vm)
    monkeypatch.setenv("FAKE_EXIT", "3")
    with pytest.raises(crusoe.CrusoeError):      # wait_ssh cannot reach the box when ssh fails
        crusoe.run(vm, "true")


def test_cli_status_and_types(fake, capsys):
    vm = crusoe.VM(name="t6")
    assert crusoe.main(["status"]) == 0
    assert "absent" in capsys.readouterr().out
    crusoe.start(vm)
    assert crusoe.main(["types"]) == 0
    assert "a100-80gb.1x" in capsys.readouterr().out


def test_missing_cli_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(crusoe.CrusoeError, match="not installed"):
        crusoe.get(crusoe.VM(name="t7"))


def test_state_accepts_both_spellings():
    assert crusoe.state({"state": "STATE_RUNNING"}) == "running"
    assert crusoe.state({"state": "STOPPED"}) == "stopped"
