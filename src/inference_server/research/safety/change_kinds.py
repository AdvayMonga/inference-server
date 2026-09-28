"""What each loop-startable change kind may write. Deny by default: unlisted paths are protected."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

ENGINE = ("src/inference_server/*",)


@dataclass(frozen=True)
class Kind:
    write: tuple[str, ...]            # may add, modify or delete
    add_only: tuple[str, ...] = ()    # may create new files only


# Only these kinds may be started by the loop; instrument/method/test/infra stay human-only.
KINDS: dict[str, Kind] = {
    "perf":     Kind(ENGINE),
    "fix":      Kind(ENGINE, add_only=("tests/test_*.py",)),
    "refactor": Kind(ENGINE),
    "obs":      Kind(ENGINE),
}

# Denied for every kind, even inside an allowed glob: the evaluator and import-time hooks.
ALWAYS_DENY = (
    "src/inference_server/research/*",
    "*conftest.py",
    "*.pth",
    "*sitecustomize.py",
    "*usercustomize.py",
)

# Removed from the agent's workspace; it can neither read nor recreate them.
HIDDEN = ("corpus/*/heldout.jsonl",)


def may_write(kind: str, path: str, *, new_file: bool) -> bool:
    """Whether a `kind` change may touch repo-relative `path`."""
    if kind not in KINDS:
        raise ValueError(f"the loop may not start a {kind!r} change")
    if any(fnmatch(path.lower(), p.lower()) for p in ALWAYS_DENY + HIDDEN):   # macOS ignores case
        return False
    k = KINDS[kind]
    if any(fnmatch(path, p) for p in k.write):
        return True
    return new_file and any(fnmatch(path, p) for p in k.add_only)
