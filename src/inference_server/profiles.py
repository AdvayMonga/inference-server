"""Launch profiles: env settings per workload class, read from profiles.toml next to this file.

[general] applies to every launch; the table named by ENGINE_PROFILE (a regime name) is applied over it. They are
written into the environment before settings load, so they override the launch env.
"""

import os
import tomllib
from pathlib import Path

PATH = Path(__file__).with_name("profiles.toml")
ENV = "ENGINE_PROFILE"


def load(path: Path = PATH) -> dict:
    """The profiles file as a dict; {} when there is none."""
    return tomllib.loads(path.read_text()) if path.exists() else {}


def resolve(name: str | None, table: dict) -> dict[str, str]:
    """[general] with [name] over it, as env strings. An unknown name is an error, not a silent general launch."""
    out = {k: str(v) for k, v in (table.get("general") or {}).items()}
    if name:
        if name not in table:
            raise ValueError(f"{ENV}={name!r}: no [{name}] table in {PATH.name}")
        out.update({k: str(v) for k, v in table[name].items()})
    return out


def apply(environ=os.environ, path: Path = PATH) -> dict[str, str]:
    """Write the selected profile into `environ`; returns what was set."""
    values = resolve(environ.get(ENV) or None, load(path))
    environ.update(values)
    return values
