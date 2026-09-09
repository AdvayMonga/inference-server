#!/usr/bin/env bash
# Launch a Modal instrument with correct provenance.
#
# Panels are emitted inside an ephemeral container that has no git repo, so the engine SHA and
# run_group must come from here. Arms of one experiment MUST share RESEARCH_RUN_GROUP — that is
# what makes them comparable; compare.py refuses across groups.
#
#   scripts/run_instrument.sh scripts/bench/bench_serving_modal.py
#   RESEARCH_RUN_GROUP=exp-42 scripts/run_instrument.sh scripts/bench/bench_serving_modal.py   # arm A
#   RESEARCH_RUN_GROUP=exp-42 scripts/run_instrument.sh scripts/bench/bench_serving_modal.py   # arm B
set -euo pipefail
cd "$(dirname "$0")/.."

export RESEARCH_ENGINE_SHA="${RESEARCH_ENGINE_SHA:-$(git rev-parse --short HEAD)}"
export RESEARCH_ENGINE_DIRTY="${RESEARCH_ENGINE_DIRTY:-$([ -n "$(git status --porcelain)" ] && echo 1 || echo 0)}"
export RESEARCH_RUN_GROUP="${RESEARCH_RUN_GROUP:-grp-$(date +%Y%m%d)-$(openssl rand -hex 3)}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Worktrees have no venv of their own; fall back to the main checkout's.
VENV="${VENV:-}"
if [ -z "$VENV" ]; then
  if [ -x "venv/bin/modal" ]; then
    VENV="venv"
  else
    MAIN_WT="$(git rev-parse --path-format=absolute --git-common-dir)/.."
    [ -x "$MAIN_WT/venv/bin/modal" ] && VENV="$MAIN_WT/venv"
  fi
fi
[ -x "$VENV/bin/modal" ] || { echo "[error] no modal venv found; set VENV=/path/to/venv" >&2; exit 1; }

echo "[provenance] sha=$RESEARCH_ENGINE_SHA dirty=$RESEARCH_ENGINE_DIRTY group=$RESEARCH_RUN_GROUP"
[ "$RESEARCH_ENGINE_DIRTY" = "1" ] && echo "[warn] working tree is dirty — this panel cannot be attributed to a clean commit"
exec "$VENV/bin/modal" run --detach "$@"
