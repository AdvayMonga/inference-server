"""lab.providers.verda against an in-memory fake of Verda's REST API."""

from __future__ import annotations

import os

import pytest

from lab import vm as labvm
from lab.providers import ProviderError, verda


class FakeApi:
    """Enough of /oauth2/token, /instances, /ssh-keys and /instance-availability to drive the adapter."""

    def __init__(self):
        self.instances: list[dict] = []
        self.keys: list[dict] = []
        self.calls: list[tuple] = []
        self.tokens = 0

    def __call__(self, method, path, body=None, auth=True):
        self.calls.append((method, path, body))
        if path == "/oauth2/token":
            self.tokens += 1
            return {"access_token": f"tok{self.tokens}", "expires_in": 3600}
        if path == "/instances" and method == "GET":
            return list(self.instances)
        if path == "/instances" and method == "POST":
            self.instances.append({"id": f"i{len(self.instances)}", "hostname": body["hostname"],
                                   "status": "provisioning", "ip": None, "instance_type": body["instance_type"],
                                   "created_at": f"2026-10-0{len(self.instances) + 1}", "location": "FIN-01",
                                   "price_per_hour": 3.7, "pricing": "DYNAMIC_PRICE"})
            return None
        if path == "/instances" and method == "PUT":
            inst = next(i for i in self.instances if i["id"] == body["id"])
            inst["status"] = {"hibernate": "hibernating", "restore": "running", "start": "running"}[body["action"]]
            inst["ip"] = "1.2.3.4" if inst["status"] == "running" else None
            return None
        if path == "/ssh-keys" and method == "GET":
            return list(self.keys)
        if path == "/ssh-keys" and method == "POST":
            self.keys.append({"id": "k1", "name": body["name"], "key": body["key"]})
            return "k1"
        if path == "/instance-availability":
            return [{"location_code": "FIN-01", "availabilities": ["1H100.80S.30V"]}]
        raise AssertionError((method, path))

    def finish_provisioning(self):
        for i in self.instances:
            if i["status"] == "provisioning":
                i["status"], i["ip"] = "running", "1.2.3.4"


@pytest.fixture
def api(monkeypatch, tmp_path):
    fake = FakeApi()
    monkeypatch.setattr(verda.Verda, "_request", lambda self, *a, **k: fake(*a, **k))
    monkeypatch.setenv("VERDA_CLIENT_ID", "id")
    monkeypatch.setenv("VERDA_CLIENT_SECRET", "secret")
    monkeypatch.setenv("LAB_VM_PROVIDER", "verda")
    key = tmp_path / "key.pub"
    key.write_text("ssh-ed25519 AAAAtest me@mac\n")
    monkeypatch.setenv("LAB_VM_KEYFILE", str(key))
    monkeypatch.setattr(labvm, "KNOWN_HOSTS_DIR", tmp_path / "kh")
    monkeypatch.setattr(labvm, "POLL_S", 0.0)
    monkeypatch.setattr(labvm, "START_TIMEOUT_S", 0.2)
    return fake


def test_create_uploads_the_local_key_once_and_deploys(api):
    p = verda.Verda()
    keyfile = os.environ["LAB_VM_KEYFILE"]
    p.create("lab-gpu", keyfile)
    p.create("lab-gpu2", keyfile)
    posts = [c for c in api.calls if c[0] == "POST" and c[1] == "/ssh-keys"]
    assert len(posts) == 1 and posts[0][2]["key"].startswith("ssh-ed25519 AAAAtest")
    deploy = next(c[2] for c in api.calls if c[0] == "POST" and c[1] == "/instances")
    assert deploy["instance_type"] == "1H100.80S.30V" and deploy["location_code"] == "FIN-01"
    assert deploy["ssh_key_ids"] == ["k1"] and deploy["os_volume"]["size"] == 200


def test_states_map_and_newest_live_instance_wins(api):
    p = verda.Verda()
    api.instances = [
        {"id": "old", "hostname": "lab-gpu", "status": "discontinued", "created_at": "2026-01-01"},
        {"id": "new", "hostname": "lab-gpu", "status": "hibernating", "created_at": "2026-02-01",
         "instance_type": "1H100.80S.30V", "ip": None},
    ]
    rec = p.get("lab-gpu")
    assert rec.state == "stopped" and rec.ip is None and "hibernating" in rec.detail
    p.start("lab-gpu")
    assert api.calls[-1][2] == {"action": "restore", "id": "new"}
    assert p.get("lab-gpu").state == "running"


def test_stop_hibernates_never_shuts_down(api):
    p = verda.Verda()
    api.instances = [{"id": "x", "hostname": "lab-gpu", "status": "running", "created_at": "1", "ip": "1.2.3.4"}]
    p.stop("lab-gpu")
    assert api.calls[-1][2]["action"] == "hibernate"
    assert all(c[2] is None or c[2].get("action") != "shutdown" for c in api.calls)


def test_start_waits_for_provisioning_then_ssh_target(api, monkeypatch):
    vm = labvm.VM(name="lab-gpu")
    api_calls_before = len(api.calls)
    # provisioning completes on the second poll
    polls = {"n": 0}
    original_get = verda.Verda.get
    def get(self, name):
        polls["n"] += 1
        if polls["n"] >= 2:
            api.finish_provisioning()
        return original_get(self, name)
    monkeypatch.setattr(verda.Verda, "get", get)
    record = labvm.start(vm)
    assert record.state == "running" and labvm.ssh_target(vm, record) == "root@1.2.3.4"
    assert len(api.calls) > api_calls_before


def test_no_capacity_is_reported_not_waited_out(api):
    vm = labvm.VM(name="lab-gpu")
    api.instances = [{"id": "x", "hostname": "lab-gpu", "status": "no_capacity", "created_at": "1",
                      "instance_type": "1H100.80S.30V", "ip": None}]
    with pytest.raises(ProviderError, match="no_capacity"):
        labvm.start(vm)


def test_running_without_ip_is_waited_for(api):
    api.instances = [{"id": "x", "hostname": "lab-gpu", "status": "running", "created_at": "1",
                      "instance_type": "1H100.80S.30V", "ip": None}]
    polls = {"n": 0}
    original = verda.Verda.get
    def get(self, name):
        polls["n"] += 1
        if polls["n"] >= 3:
            api.instances[0]["ip"] = "5.6.7.8"
        return original(self, name)
    api_get = get
    import unittest.mock as um
    with um.patch.object(verda.Verda, "get", api_get):
        record = labvm.start(labvm.VM(name="lab-gpu"))
    assert record.ip == "5.6.7.8" and polls["n"] >= 3


def test_stop_refuses_offline_with_advice(api):
    api.instances = [{"id": "x", "hostname": "lab-gpu", "status": "offline", "created_at": "1",
                      "instance_type": "1H100.80S.30V", "ip": None}]
    with pytest.raises(ProviderError, match="start.*then.*stop"):
        labvm.stop(labvm.VM(name="lab-gpu"))
    assert not any(c[0] == "PUT" for c in api.calls)


def test_bad_payloads_are_provider_errors(api, monkeypatch):
    monkeypatch.setattr(verda.Verda, "_request", lambda self, *a, **k: None)
    with pytest.raises(ProviderError, match="expected a list"):
        verda.Verda().get("lab-gpu")


def test_missing_keyfile_is_a_provider_error(api, monkeypatch):
    with pytest.raises(ProviderError, match="LAB_VM_KEYFILE"):
        verda.Verda().create("lab-gpu", "/nonexistent/key.pub")


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("VERDA_CLIENT_ID", raising=False)
    monkeypatch.delenv("VERDA_CLIENT_SECRET", raising=False)
    with pytest.raises(ProviderError, match="VERDA_CLIENT_ID"):
        verda.Verda()._access_token()


def test_token_is_cached(api):
    p = verda.Verda()
    p._access_token(); p._access_token()
    assert api.tokens == 1
