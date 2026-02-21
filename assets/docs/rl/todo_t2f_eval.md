## TODO: t2f-eval and Rerun Logging

### Idea

Custom eval/training visualization with Rerun instead of Rich TUI.

### Two Approaches

**1. Eval-level: `t2f-eval` command**

Verifiers' `run_evaluation()` in `eval_utils.py` takes `on_progress` and `on_log` callbacks. The Rich TUI is just one implementation. Write a `t2f-eval` that passes Rerun callbacks instead. Fires on every completed rollout — live logging, not post-processing.

**2. Environment-level: callback on `env_response`**

Log from inside the multi-turn loop via an optional callback on the environment. No Rerun dependency in the environment itself — keep it portable for the hub:

```python
class MyEnv(vf.MultiTurnEnv):
    def __init__(self, on_step=None, **kwargs):
        self.on_step = on_step
        super().__init__(**kwargs)

    async def env_response(self, messages, state):
        result = do_stuff(messages)
        if self.on_step:
            self.on_step(messages, state, result)
        return result
```

This gives you per-turn data in real time: model actions, observations, trajectory, turn count, info. Scores are 1 step behind (computed after the full rollout), but everything else is live.

### Tradeoffs

- **Eval-level** — sees final scores and metrics per rollout, no verifiers changes needed, but no per-turn visibility
- **Environment-level** — sees every turn in real time, but only the data available at that point (no final score yet). Keeps Rerun out of the environment via optional callback
- Both can work together — per-turn logging in `env_response`, final scores via eval callback
