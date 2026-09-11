## What changed


## Why


## Evidence for the engine change
<!-- A diff under src/inference_server/ (not research/, not static/) needs ONE of these
     committed on this branch, or premerge_check.py refuses the merge. Prose here is not one
     of them. Each record vouches for ONE commit, with no engine change after it.
     Delete the rows that don't apply; see CONTRIBUTING.md.

  faster        A/B experiment, five green gates. Do NOT rebase after measuring.
  bug fix       a record with regression_test + engine_sha_base. The gate RE-RUNS the test;
                confirm it fails with the fix reverted.
  no behaviour  loop no-claim --why "..." --sha <sha>   (rename, dead import, comment)

     Touched no engine file? Say so — the gate passes on its own. -->


## Verification
<!-- What you actually ran, and what it does NOT cover. "pytest -q" says nothing about GPU
     paths: the model-heavy tests are deselected and no CI lane can launch a kernel. If this
     touches src/inference_server/models/, dispatch the gpu lane. -->


## Docs
<!-- A change to a component, endpoint, env var, metric or data-flow path updates
     docs/architecture.html AND the relevant arch-*.html, same PR. -->
