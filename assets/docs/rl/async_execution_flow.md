## Async Execution Flow in Verifiers

How `env.generate()` orchestrates groups, rollouts, scoring, and callbacks.

### The Flow

1. **`env.generate()` is called** with `n` examples and `r` rollouts per example.

2. **Inputs are built** — each example gets `r` copies, so 10 examples × 3 rollouts = 30 total inputs.

3. **Inputs are grouped by `example_id`** — all 3 rollouts for the same prompt end up in one group. This is critical because GRPO needs the group to compute advantages.

4. **Each group becomes an `asyncio.create_task`** — groups run in parallel. A semaphore controls how many groups run concurrently (prevents overwhelming the API/vLLM):

```
Group 0 (prompt A, 3 rollouts) ──┐
Group 1 (prompt B, 3 rollouts) ──┼── all launched as tasks
Group 2 (prompt C, 3 rollouts) ──┘
                                  semaphore limits concurrency
```

5. **Inside each group (`run_group`)** — the rollouts within a group also run in parallel. Each rollout is its own async call to the model. For multi-turn, each rollout runs the full loop (prompt → model → env_response → model → ... → stop) independently.

6. **Rubric scores the group** — once all rollouts in a group finish, the rubric runs. It scores all rollouts, computes per-rollout rewards, metrics, and group-relative advantages. This happens inside `run_group`, before it returns.

7. **`asyncio.as_completed` picks up finished groups** — as each group finishes (rollouts done + rubric scored), it yields. The order is whichever group finishes first, not input order.

8. **`on_progress` fires per completed group** — right after a group is yielded by `as_completed`. This is where the TUI updates, and where a Rerun logger would hook in. At this point, the state has everything: trajectory, completion, reward, metrics, advantage.

9. **Results are saved incrementally** — after `on_progress`, results for that group are appended to the output file. You don't wait for all groups to finish.

10. **After all groups complete** — final summary, return all results.

### What's Parallel, What's Sequential

| Level | Parallel? | How |
|-------|-----------|-----|
| Groups (different prompts) | Yes | `asyncio.create_task` per group, semaphore-bounded |
| Rollouts within a group | Yes | All rollouts for same prompt run concurrently |
| Turns within a rollout | Sequential | Must wait for model → env_response → model |
| Rubric scoring | Sequential per group | Runs after all rollouts in group finish |
| `on_progress` | Sequential | Fires one group at a time as they complete |

### Timeline Example (3 prompts, 2 rollouts each)

```
t=0   Launch Group0(A), Group1(B), Group2(C)
t=1   Group0: rollout0 turn1, rollout1 turn1  (parallel)
      Group1: rollout0 turn1, rollout1 turn1  (parallel)
      Group2: rollout0 turn1, rollout1 turn1  (parallel)
t=2   Group1 finishes first → rubric scores → on_progress → save
t=3   Group0 finishes → rubric scores → on_progress → save
t=4   Group2 finishes → rubric scores → on_progress → save
t=5   All done, return results
```

### Key Insight

Scoring and progress happen at the group boundary. You never get a score mid-rollout or mid-group. A group is the atomic unit — all its rollouts must finish before anything gets scored or reported.
