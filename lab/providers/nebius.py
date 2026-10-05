"""Nebius AI Cloud through the `nebius` CLI (auth: `nebius profile create`). A stopped VM keeps its boot disk; GPU billing ends."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lab.providers import ProviderError, Record, env_field

USER = "lab"   # created by cloud-init; Nebius refuses root and admin as login names


@dataclass(frozen=True)
class Nebius:
    type: str = env_field("LAB_VM_TYPE", "1gpu-16vcpu-200gb")          # a preset of `platform`
    platform: str = env_field("LAB_VM_PLATFORM", "gpu-h200-sxm")
    location: str = env_field("LAB_VM_LOCATION", "")                   # the CLI profile's project decides; informational
    image: str = env_field("LAB_VM_IMAGE", "ubuntu24.04-cuda13.0")      # an image family; CUDA 13 driver, as on Verda (setup installs cu130 torch)
    disk_gb: str = env_field("LAB_VM_DISK_GB", "400")
    project: str = env_field("LAB_VM_PROJECT", "")                      # --parent-id; empty uses the profile's
    subnet: str = env_field("LAB_VM_SUBNET", "")                        # empty: the project's first subnet
    default_user: str = USER
    stop_word: str = "stop"

    def _cli(self, *args: str) -> str:
        argv = ["nebius", *args]
        if self.project and args[:1] in (["compute"], ["vpc"]) and args[2:3] in (["list"], ["create"]):
            argv += ["--parent-id", self.project]
        try:
            out = subprocess.run(argv, capture_output=True, text=True)
        except FileNotFoundError:
            raise ProviderError("`nebius` CLI not installed: https://docs.nebius.com/cli/install")
        if out.returncode != 0:
            raise ProviderError(f"nebius {' '.join(args[:4])}: {out.stderr.strip() or out.stdout.strip()}")
        return out.stdout

    def _json(self, *args: str):
        text = self._cli(*args, "--format", "json")
        try:
            return json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            raise ProviderError(f"nebius {' '.join(args[:4])}: not JSON: {text[:200]}")

    def _raw(self, name: str) -> dict | None:
        items = self._json("compute", "instance", "list").get("items") or []
        return next((i for i in items if (i.get("metadata") or {}).get("name") == name), None)

    def get(self, name: str) -> Record | None:
        raw = self._raw(name)
        if raw is None:
            return None
        status = raw.get("status") or {}
        state = str(status.get("state", "")).lower()
        ip = None
        for nic in status.get("network_interfaces") or []:
            addr = (nic.get("public_ip_address") or {}).get("address")
            if addr:
                ip = addr.split("/")[0]
                break
        res = (raw.get("spec") or {}).get("resources") or {}
        kind = f"{res.get('platform', '')}/{res.get('preset', '')}".strip("/")
        return Record(name=name, state=state, ip=ip, type=kind, detail=state)

    def _subnet(self) -> str:
        if self.subnet:
            return self.subnet
        items = self._json("vpc", "subnet", "list").get("items") or []
        if not items:
            raise ProviderError("no VPC subnet in this project; set LAB_VM_SUBNET")
        return items[0]["metadata"]["id"]

    def create(self, name: str, keyfile: str) -> None:
        try:
            public = Path(os.path.expanduser(keyfile)).read_text().strip()
        except OSError as e:
            raise ProviderError(f"ssh public key {keyfile}: {e.strerror}; set LAB_VM_KEYFILE")
        user_data = ("#cloud-config\nusers:\n"
                     f"  - name: {USER}\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n"
                     f"    ssh_authorized_keys:\n      - {public}\n")
        nics = json.dumps([{"name": "eth0", "ip_address": {}, "public_ip_address": {}, "subnet_id": self._subnet()}])
        self._cli("compute", "instance", "create", "--name", name,
                  "--resources-platform", self.platform, "--resources-preset", self.type,
                  "--boot-disk-managed-disk-name", f"{name}-boot",
                  "--boot-disk-managed-disk-type", "network_ssd",
                  "--boot-disk-managed-disk-size-gibibytes", str(self.disk_gb),
                  "--boot-disk-managed-disk-source-image-family-image-family", self.image,
                  "--boot-disk-attach-mode", "READ_WRITE",
                  "--network-interfaces", nics, "--cloud-init-user-data", user_data)

    def _id(self, name: str) -> str:
        raw = self._raw(name)
        if raw is None:
            raise ProviderError(f"{name}: no instance")
        return raw["metadata"]["id"]

    def start(self, name: str) -> None:
        self._cli("compute", "instance", "start", self._id(name))

    def stop(self, name: str) -> None:
        self._cli("compute", "instance", "stop", self._id(name))

    def delete(self, name: str) -> None:
        """Instance and its managed boot disk; `lab.vm` never calls this, it is for ending for good."""
        self._cli("compute", "instance", "delete", self._id(name))

    def types(self) -> str:
        items = self._json("compute", "platform", "list").get("items") or []
        lines = []
        for p in items:
            presets = [x.get("name") for x in (p.get("spec") or {}).get("presets") or []]
            lines.append(f"{(p.get('metadata') or {}).get('name')}: {', '.join(filter(None, presets))}")
        return "\n".join(lines) + "\n"
