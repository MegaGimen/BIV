# SWE bash+file scaffold note

Meta’s Muse Glimmer methodology describes SWE-Bench Verified/Pro with an
internal agent that exposes **a bash tool and a file view/create/edit tool**.

Harbor does not ship that exact Meta agent. This tree intentionally **does not**
reimplement a full custom Harbor agent yet.

**Default in `test.py`:** Harbor’s built-in `mini-swe-agent` (bash-centric, thin
control flow) — closest widely used public scaffold for model-vs-model SWE
comparisons.

If we later need a stricter Meta clone, add a Harbor-compatible agent package
here and point `eval/run_harbor.py` `SUITES[*].agent` at its import path.
