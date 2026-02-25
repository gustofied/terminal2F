## Integrating External Systems with Verifiers

Verifiers doesn't care what your environment does internally. It provides a lifecycle via `MultiTurnEnv` - you hook into it, do whatever you want behind the hooks, and return messages.

### The Hooks

**`setup_state(state) -> State`**

Called once at the start of each rollout. Use it to spin up your external system and store handles in `state`.

```python
async def setup_state(self, state: vf.State) -> vf.State:
    db = create_world(seed=state["info"]["seed"])
    server = start_mcp_server(db)
    state["db"] = db
    state["server"] = server
    return state
```

**`env_response(messages, state) -> Messages`**

Called after every model turn. You receive the conversation so far, do your thing (execute tool calls, query a database, call an API), and return the next messages.

This is the main integration point. Verifiers doesn't know or care what happens inside - it just expects messages back.

```python
async def env_response(self, messages, state, **kwargs):
    last_msg = messages[-1]
    tool_calls = parse_tool_calls(last_msg)
    results = []
    for call in tool_calls:
        result = await state["server"].call_tool(call.name, call.args)
        results.append({"role": "tool", "content": result, "tool_call_id": call.id})
    return results
```

**`@vf.stop`**

Custom stop conditions. Checked after every turn. Return `True` to end the rollout.

```python
@vf.stop
async def workflow_complete(self, state: vf.State) -> bool:
    return state.get("done", False)
```

The built-in `max_turns_reached` stop is always available. Add your own for domain-specific conditions.

**`@vf.cleanup`**

Called after each rollout completes. Tear down per-rollout resources.

```python
@vf.cleanup
async def cleanup(self, state: vf.State) -> None:
    server = state.pop("server", None)
    if server:
        await server.stop()
```

**`@vf.teardown`**

Called once when the entire environment shuts down. Clean up global resources.

```python
@vf.teardown
async def teardown(self) -> None:
    # clean up anything shared across rollouts
    pass
```

### The State Dict

`state` is the glue. It persists across turns within a rollout and is accessible everywhere - in `env_response`, stop conditions, cleanup, and rubric scoring functions.

Key fields managed by verifiers:
- `state["prompt"]` - initial prompt messages
- `state["completion"]` - final computed completion
- `state["trajectory"]` - list of `{prompt, completion, reward, ...}` per turn
- `state["info"]` - per-example data from your dataset (ground truths, seeds, config)
- `state["error"]` - any error that occurred

You add whatever you need:
- `state["db"]` - database connection
- `state["server"]` - MCP server handle
- `state["done"]` - custom done flag

### Rubric / Scoring

Rubric functions receive `completion` and `state`. Use `state` to access ground truths, database state, or anything else you stored during the rollout.

```python
async def score(completion, state, **kwargs):
    expected = state["info"]["ground_truths"]
    actual = query_final_state(state["db"])
    return compute_score(actual, expected)

rubric = vf.Rubric(funcs=[score], weights=[1.0])
```

### How Real Integrations Use This

**OpenEnv** - spins up a sandboxed server in `setup_state`, routes tool calls through MCP in `env_response`, tracks intermediate rewards in `state["trajectory"]`, cleans up sandbox in `@vf.cleanup`.

**Harbor** - loads task files in `__init__`, uploads instructions in `post_sandbox_setup`, runs verification tests in `post_rollout` to compute reward. Doesn't even override `env_response`.

**Email-to-CC-BCC** - stores follow-up prompts and ground truths in `state["info"]`, returns the next email in `env_response`, scores final state with set overlap in the rubric.

All completely different external systems. Same four hooks.

### Minimal Template

```python
import verifiers as vf
from datasets import Dataset

class MyEnv(vf.MultiTurnEnv):
    @vf.stop
    async def done(self, state: vf.State) -> bool:
        return state.get("done", False)

    async def setup_state(self, state: vf.State) -> vf.State:
        # start your external system, store in state
        return state

    async def env_response(self, messages, state, **kwargs):
        # handle model output, return next messages
        return [{"role": "user", "content": "next prompt"}]

    @vf.cleanup
    async def cleanup(self, state: vf.State) -> None:
        # tear down per-rollout resources
        pass

def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_list([...])
    rubric = vf.Rubric(funcs=[...], weights=[...])
    return MyEnv(dataset=dataset, rubric=rubric, max_turns=10)
```
