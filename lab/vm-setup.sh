#!/usr/bin/env bash
# One-time bootstrap of a fresh Crusoe VM for the lab. Idempotent; run via `python -m lab.crusoe setup`.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== uv + venv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra dev
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA wheel; pyproject pins CPU on linux

echo "== Nsight"
if ! command -v nsys >/dev/null || ! command -v ncu >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq nsight-systems nsight-compute 2>/dev/null \
    || echo "Nsight not in apt; install from the CUDA toolkit repo (finding, not failure)"
fi
nsys --version 2>/dev/null || true
ncu --version 2>/dev/null | head -1 || true

echo "== counters: can a non-admin process read GPU performance counters?"
# ncu needs NVreg_RestrictProfilingToAdminUsers=0 (or root). Record the answer; the lab treats it as a fact about this venue.
if [ -r /proc/driver/nvidia/params ]; then
  grep -o 'RestrictProfilingToAdminUsers: [0-9]' /proc/driver/nvidia/params || echo "RestrictProfilingToAdminUsers: unknown"
fi

echo "== clocks: can we lock them?"
sudo nvidia-smi -pm 1 >/dev/null 2>&1 && echo "persistence mode: on" || echo "persistence mode: refused"
sudo nvidia-smi --query-gpu=clocks.max.sm,clocks.applications.graphics --format=csv,noheader

echo "== model cache"
mkdir -p "$HOME/.cache/huggingface"
du -sh "$HOME/.cache/huggingface" 2>/dev/null || true

echo "== smoke"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "setup done"
