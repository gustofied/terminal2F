## TODO: t2f-eval

### Idea

Custom eval runner with Rerun logging instead of the Rich TUI that `vf-eval` and `prime eval` use.

### Why

`vf-eval` and `prime eval` hardcode their display logic (Rich TUI / progress bars). No way to plug in custom logging through the CLI. But verifiers' `run_evaluation()` in `eval_utils.py` already takes `on_progress` and `on_log` callbacks — the TUI is just one implementation of those callbacks.

### How

Write a `t2f-eval` command that calls `run_evaluation()` directly with Rerun callbacks instead of Rich ones. Same eval logic, different visualization. No verifiers changes needed.

The callbacks fire on every completed rollout, so logging is live — not a post-processing step.
