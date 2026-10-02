"""Crusoe Cloud through the `crusoe` CLI (auth: `crusoe config init`). A stopped on-demand VM keeps its disk and bills nothing."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

from lab.providers import ProviderError, Record, env_field


@dataclass(frozen=True)
class Crusoe:
    type: str = env_field("LAB_VM_TYPE", "a100-80gb.1x")
    location: str = env_field("LAB_VM_LOCATION", "us-east1-a")
    image: str = env_field("LAB_VM_IMAGE", "ubuntu22.04-nvidia-slurm:latest")
    default_user: str = "ubuntu"
    stop_word: str = "stop"

    def _cli(self, *args: str) -> str:
        try:
            out = subprocess.run(["crusoe", *args], capture_output=True, text=True)
        except FileNotFoundError:
            raise ProviderError("`crusoe` CLI not installed: https://docs.crusoecloud.com/quickstart/installing-the-cli/")
        if out.returncode != 0:
            raise ProviderError(f"crusoe {' '.join(args)}: {out.stderr.strip() or out.stdout.strip()}")
        return out.stdout

    def get(self, name: str) -> Record | None:
        listing = json.loads(self._cli("compute", "vms", "list", "--json"))
        if isinstance(listing, dict):                      # some CLI versions wrap the array
            listing = listing.get("items") or listing.get("vms") or []
        raw = next((r for r in listing if r.get("name") == name), None)
        if raw is None:
            return None
        state = str(raw.get("state", "")).lower().removeprefix("state_")
        ip = next((a["public_ipv4"]["address"] for nic in raw.get("network_interfaces", [])
                   for a in nic.get("ips", []) if (a.get("public_ipv4") or {}).get("address")), None)
        return Record(name=name, state=state, ip=ip, type=raw.get("type", ""), detail=state)

    def create(self, name: str, keyfile: str) -> None:
        self._cli("compute", "vms", "create", "--name", name, "--type", self.type, "--location",
                  self.location, "--image", self.image, "--keyfile", os.path.expanduser(keyfile))

    def start(self, name: str) -> None:
        self._cli("compute", "vms", "start", name)

    def stop(self, name: str) -> None:
        self._cli("compute", "vms", "stop", name)

    def types(self) -> str:
        return self._cli("compute", "vms", "types")
