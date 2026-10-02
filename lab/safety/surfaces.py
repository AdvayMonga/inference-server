"""What the agent may write. Deny by default: unlisted paths are protected."""

from __future__ import annotations

from fnmatch import fnmatch

WRITE = ("src/inference_server/*",)          # may add, modify or delete
ADD_ONLY = ("tests/test_*.py",)              # new tests add coverage, never evidence
# Denied even inside an allowed glob: the evaluator's own code and import-time hooks.
ALWAYS_DENY = (
    "*conftest.py", "*.pth", "*sitecustomize.py", "*usercustomize.py",
    "*ruff.toml", "*pyproject.toml", "*setup.cfg", "*pytest.ini", "*tox.ini",
)
# Removed from the agent's workspace; it can neither read nor recreate them.
HIDDEN = ("corpus/*/heldout.jsonl",)


def may_write(path: str, *, new_file: bool) -> bool:
    """Whether the agent may touch repo-relative `path`."""
    if any(fnmatch(path.lower(), p.lower()) for p in ALWAYS_DENY + HIDDEN):   # macOS ignores case
        return False
    if any(fnmatch(path, p) for p in WRITE):
        return True
    return new_file and any(fnmatch(path, p) for p in ADD_ONLY)
