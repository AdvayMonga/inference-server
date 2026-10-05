"""lab.providers.nebius against a fake `nebius` CLI on PATH: lifecycle, ip parsing, cloud-init key."""

from __future__ import annotations

import json
import os
import stat

import pytest

from lab import vm as labvm
from lab.providers import ProviderError, nebius

FAKE = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
open(os.environ["FAKE_LOG"], "a").write(json.dumps(args) + "\n")
p = os.environ["FAKE_STATE"]
db = json.load(open(p)) if os.path.exists(p) else {}
def save(): json.dump(db, open(p, "w"))
def inst(n, v):
    return {"metadata": {"id": v["id"], "name": n},
            "spec": {"resources": {"platform": "gpu-h200-sxm", "preset": "1gpu-16vcpu-200gb"}},
            "status": {"state": v["state"], "network_interfaces": [{"public_ip_address": {"address": "203.0.113.9/32"}}]}}
if args[:3] == ["compute", "instance", "list"]:
    print(json.dumps({"items": [inst(n, v) for n, v in db.items()]}))
elif args[:3] == ["compute", "instance", "create"]:
    n = args[args.index("--name") + 1]
    db[n] = {"id": "computeinstance-" + n, "state": "RUNNING", "user_data": args[args.index("--cloud-init-user-data") + 1]}; save()
elif args[:3] in (["compute", "instance", "stop"], ["compute", "instance", "start"], ["compute", "instance", "delete"]):
    n = next(k for k, v in db.items() if v["id"] == args[3])
    if args[2] == "delete": db.pop(n)
    else: db[n]["state"] = "STOPPED" if args[2] == "stop" else "RUNNING"
    save()
elif args[:3] == ["vpc", "subnet", "list"]:
    print(json.dumps({"items": [{"metadata": {"id": "vpcsubnet-1"}}]}))
elif args[:3] == ["compute", "platform", "list"]:
    print(json.dumps({"items": [{"metadata": {"name": "gpu-h200-sxm"}, "spec": {"presets": [{"name": "1gpu-16vcpu-200gb"}]}}]}))
else:
    sys.stderr.write("unknown " + " ".join(args)); sys.exit(2)
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    f = b / "nebius"
    f.write_text(FAKE)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("LAB_VM_PROVIDER", "nebius")
    key = tmp_path / "key.pub"
    key.write_text("ssh-ed25519 AAAAneb lab@mac\n")
    monkeypatch.setenv("LAB_VM_KEYFILE", str(key))
    monkeypatch.setattr(labvm, "KNOWN_HOSTS_DIR", tmp_path / "kh")
    monkeypatch.setattr(labvm, "POLL_S", 0.0)
    return tmp_path


def calls(tmp):
    return [json.loads(line) for line in (tmp / "calls.log").read_text().splitlines()]


def test_create_runs_with_h200_preset_cloud_init_key_and_subnet(fake):
    rec = labvm.start(labvm.VM(name="g"))
    assert rec.state == "running" and rec.ip == "203.0.113.9" and rec.type == "gpu-h200-sxm/1gpu-16vcpu-200gb"
    create = next(c for c in calls(fake) if c[:3] == ["compute", "instance", "create"])
    assert create[create.index("--resources-platform") + 1] == "gpu-h200-sxm"
    assert create[create.index("--resources-preset") + 1] == "1gpu-16vcpu-200gb"
    assert create[create.index("--boot-disk-managed-disk-size-gibibytes") + 1] == "400"
    assert json.loads(create[create.index("--network-interfaces") + 1])[0]["subnet_id"] == "vpcsubnet-1"
    ud = json.load(open(fake / "state.json"))["g"]["user_data"]
    assert ud.startswith("#cloud-config") and "name: lab" in ud and "ssh-ed25519 AAAAneb" in ud


def test_stop_keeps_the_vm_and_start_resumes_it(fake):
    vm = labvm.VM(name="g")
    labvm.start(vm)
    labvm.stop(vm)
    assert vm.provider.get("g").state == "stopped"
    labvm.start(vm)
    kinds = [c[2] for c in calls(fake) if c[:2] == ["compute", "instance"]]
    assert kinds.count("create") == 1 and kinds.count("stop") == 1 and kinds.count("start") == 1


def test_login_is_the_cloud_init_user(fake):
    vm = labvm.VM(name="g")
    assert labvm.ssh_target(vm, labvm.start(vm)) == "lab@203.0.113.9"


def test_types_lists_platform_presets(fake):
    assert "gpu-h200-sxm: 1gpu-16vcpu-200gb" in nebius.Nebius().types()


def test_delete_removes_instance(fake):
    labvm.start(labvm.VM(name="g"))
    nebius.Nebius().delete("g")
    assert nebius.Nebius().get("g") is None


def test_missing_cli_and_missing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ProviderError, match="not installed"):
        nebius.Nebius().get("g")
    with pytest.raises(ProviderError, match="LAB_VM_KEYFILE"):
        nebius.Nebius(subnet="s").create("g", "/nope/key.pub")
