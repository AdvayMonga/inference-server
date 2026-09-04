#!/usr/bin/env bash
# Launch a Modal instrument with correct provenance.
#
# Panels are emitted inside an ephemeral container that has no git repo, so the engine SHA and
# run_group must come from here. Arms of one experiment MUST share RESEARCH_RUN_GROUP — that is
# what makes them comparable; compare.py refuses across groups.
#
#   scripts/run_instrument.sh scripts/bench_serving_modal.py
#   RESEARCH_RUN_GROUP=exp-42 scripts/run_instrument.sh scripts/bench_serving_modal.py   # arm A
#   RESEARCH_RUN_GROUP=exp-42 scripts/run_instrument.sh scripts/bench_serving_modal.py   # arm B
set -euo pipefail
cd "$(dirname "$0")/.."

export RESEARCH_ENGINE_SHA="${RESEARCH_ENGINE_SHA:-$(git rev-parse --short HEAD)}"
export RESEARCH_ENGINE_DIRTY="${RESEARCH_ENGINE_DIRTY:-$([ -n "$(git status --porcelain)" ] && echo 1 || echo 0)}"
export RESEARCH_RUN_GROUP="${RESEARCH_RUN_GROUP:-grp-$(date +%Y%m%d)-$(openssl rand -hex 3)}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "[provenance] sha=$RESEARCH_ENGINE_SHA dirty=$RESEARCH_ENGINE_DIRTY group=$RESEARCH_RUN_GROUP"
[ "$RESEARCH_ENGINE_DIRTY" = "1" ] && echo "[warn] working tree is dirty — this panel cannot be attributed to a clean commit"
exec venv/bin/modal run --detach "$@"
