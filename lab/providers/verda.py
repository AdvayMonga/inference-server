"""Verda (ex DataCrunch) through its REST API (auth: VERDA_CLIENT_ID / VERDA_CLIENT_SECRET from the console).

`stop` here is Verda's hibernate: compute billing ends, the OS volume stays and is billed as storage.
Verda's own `shutdown` keeps billing the GPU, so it is never used.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from lab.providers import ProviderError, Record, as_list, env_field

API = "https://api.datacrunch.io/v1"
RUNNING, STOPPED = "running", "stopped"
# Verda statuses -> ours. Anything unlisted is transitional (provisioning, restoring, ...).
STATES = {"running": RUNNING, "hibernating": STOPPED, "offline": "offline",
          "error": "error", "installation_failed": "error", "no_capacity": "no_capacity",
          "discontinued": "absent", "deleting": "absent", "notfound": "absent"}


@dataclass(frozen=True)
class Verda:
    type: str = env_field("LAB_VM_TYPE", "1H100.80S.30V")
    location: str = env_field("LAB_VM_LOCATION", "FIN-01")
    image: str = env_field("LAB_VM_IMAGE", "ubuntu-24.04-cuda-12.8-open-docker")
    disk_gb: int = field(default_factory=lambda: int(os.environ.get("LAB_VM_DISK_GB", "200")))
    default_user: str = "root"
    stop_word: str = "hibernate"
    _token: dict = field(default_factory=dict, compare=False)

    # -- http ---------------------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None, auth: bool = True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self._access_token()}"
        req = urllib.request.Request(f"{API}{path}", method=method, headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raise ProviderError(f"verda {method} {path}: {e.code} {e.read().decode(errors='replace')[:300]}")
        except urllib.error.URLError as e:
            raise ProviderError(f"verda {method} {path}: {e.reason}")
        return json.loads(raw) if raw.strip() else None

    def _access_token(self) -> str:
        if self._token.get("expires_at", 0) > time.time() + 60:
            return self._token["access_token"]
        cid, secret = os.environ.get("VERDA_CLIENT_ID"), os.environ.get("VERDA_CLIENT_SECRET")
        if not cid or not secret:
            raise ProviderError("set VERDA_CLIENT_ID and VERDA_CLIENT_SECRET (console.verda.com > Keys); "
                                "never in the repo")
        tok = self._request("POST", "/oauth2/token", {"grant_type": "client_credentials",
                                                      "client_id": cid, "client_secret": secret},
                            auth=False)
        self._token.update(tok, expires_at=time.time() + float(tok.get("expires_in", 3600)))
        return tok["access_token"]

    # -- lifecycle ----------------------------------------------------------------------
    def _raw(self, name: str) -> dict | None:
        """Verda has no unique names; the hostname is ours, and the newest live match wins."""
        live = [i for i in as_list(self._request("GET", "/instances"), "GET /instances")
                if i.get("hostname") == name and STATES.get(i.get("status"), "") != "absent"]
        return max(live, key=lambda i: i.get("created_at", ""), default=None)

    def get(self, name: str) -> Record | None:
        raw = self._raw(name)
        if raw is None:
            return None
        status = raw.get("status", "")
        return Record(name=name, state=STATES.get(status, status), ip=raw.get("ip") or None,
                      type=raw.get("instance_type", ""),
                      detail=f"{status} {raw.get('location', '')} ${raw.get('price_per_hour', '?')}/h "
                             f"{raw.get('pricing', '')}".strip())

    def _ssh_key_id(self, keyfile: str) -> str:
        """The local public key's id at Verda, uploaded on first use."""
        try:
            public = Path(os.path.expanduser(keyfile)).read_text().strip()
        except OSError as e:
            raise ProviderError(f"ssh public key {keyfile}: {e.strerror}; set LAB_VM_KEYFILE")
        for key in as_list(self._request("GET", "/ssh-keys"), "GET /ssh-keys"):
            if key.get("key", "").split()[:2] == public.split()[:2]:
                return key["id"]
        created = self._request("POST", "/ssh-keys", {"name": "lab", "key": public})
        key_id = created if isinstance(created, str) else (created or {}).get("id")
        if not key_id:
            raise ProviderError(f"POST /ssh-keys returned no id: {created!r}")
        return key_id

    def create(self, name: str, keyfile: str) -> None:
        self._request("POST", "/instances", {
            "instance_type": self.type, "image": self.image, "hostname": name,
            "description": "inference-server lab", "location_code": self.location,
            "ssh_key_ids": [self._ssh_key_id(keyfile)],
            "os_volume": {"name": f"{name}-os", "size": self.disk_gb}})

    def _action(self, name: str, choose) -> None:
        """One listing, then the action `choose(status)` picks for that same record."""
        raw = self._raw(name)
        if raw is None:
            raise ProviderError(f"{name}: no instance")
        action = choose(raw.get("status", ""))
        self._request("PUT", "/instances", {"action": action, "id": raw["id"]})

    def start(self, name: str) -> None:
        self._action(name, lambda status: "restore" if status == "hibernating" else "start")

    def stop(self, name: str) -> None:
        """Hibernate ends GPU billing; it is only accepted from `running`, so an `offline` box is started first."""
        raw = self._raw(name)
        if raw is not None and raw.get("status") == "offline":
            raise ProviderError(f"{name} is shut down (offline), which still bills the GPU; "
                                f"run `start` then `stop` to hibernate it")
        self._action(name, lambda status: "hibernate")

    def types(self) -> str:
        avail = as_list(self._request("GET", "/instance-availability"), "GET /instance-availability")
        lines = [f"{a['location_code']}: {', '.join(a['availabilities'])}" for a in avail]
        return "\n".join(lines) + "\n"
