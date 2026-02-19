## Verifiers - Notable Changes

Things that changed in the verifiers library worth knowing about.

### @vf.stop and is_completed

`is_completed` lives on the base `Environment` class with `@final`, which means it cannot be overridden. What it does internally is loop through all methods you've decorated with `@vf.stop` and check them one by one. If any returns `True`, the rollout stops.

So instead of overriding `is_completed`, you just define your own stop conditions with `@vf.stop` and the base class picks them up automatically.

**The old way:** you'd override `is_completed` and put all your stop logic in one big method. Problem is, `MultiTurnEnv` also needs its own stops - error checking, max turns, prompt too long. If you override `is_completed`, you either lose those or have to remember to call `super()`.

**The new way:**

- The base class already registers built-in stops (errors, max turns, etc.) via `@vf.stop`
- You add your own stops via `@vf.stop`
- `is_completed` (which you can't touch) runs them all automatically

It's a plugin system basically. You just declare "here's a reason to stop" and the framework handles the rest. No risk of accidentally breaking the built-in safety checks.

### Writing stop conditions

Each `@vf.stop` is one reason to end a rollout. You can have as many as you want:

```python
@vf.stop
async def all_turns_done(self, state: State) -> bool:
    return len(state["trajectory"]) >= state["info"]["num_turns"]

@vf.stop
async def gave_up(self, state: State) -> bool:
    last = state["trajectory"][-1]
    return "I don't know" in last.get("content", "")

@vf.stop(priority=10)  # runs first
async def budget_exceeded(self, state: State) -> bool:
    return state.get("total_tokens", 0) > 4000
```

Each one is a separate, named condition. When any returns `True`, the rollout stops and `state["stop_condition"]` tells you which one fired. Makes it easy to debug and track why rollouts ended.