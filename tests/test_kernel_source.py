"""Every @triton.jit kernel must reference only names it can actually see.

Triton compiles a kernel from its source on first launch, so a name that exists only inside a
*different* function is a NameError on the GPU — and invisible to the CPU suite, which never
launches a kernel. A stray edit left `_paged_decode_kernel` referencing `NEG`, a local of the
tiled prefill kernel, for three commits. This check is static, runs without triton installed,
fails at that base sha and passes at the fix.
"""

import ast
import builtins
from pathlib import Path

KERNELS = Path(__file__).resolve().parents[1] / "src/inference_server/models/paged_attention_kernel.py"


def _module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            names.update(n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _is_jit(fn: ast.FunctionDef) -> bool:
    return any(
        (isinstance(d, ast.Attribute) and d.attr == "jit") or (isinstance(d, ast.Name) and d.id == "jit")
        for d in fn.decorator_list
    )


def _free_names(fn: ast.FunctionDef) -> set[str]:
    """Names loaded in the body that the function itself never binds."""
    args = fn.args
    bound = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    bound.update(a.arg for a in (args.vararg, args.kwarg) if a)
    loaded: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else loaded).add(node.id)
    return loaded - bound


def test_triton_kernels_reference_only_names_they_can_see():
    tree = ast.parse(KERNELS.read_text())
    visible = _module_names(tree) | set(dir(builtins))
    bad = {}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and _is_jit(fn):
            missing = _free_names(fn) - visible
            if missing:
                bad[fn.name] = sorted(missing)
    assert not bad, f"undefined names inside @triton.jit kernels: {bad}"
