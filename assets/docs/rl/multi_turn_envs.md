## Multi-Turn Environments in Verifiers

### The Pieces

**Dataset row** — the starting state. Has `prompt` (initial message), plus any extra data you need during the rollout. Extra data goes in `info`:

```python
{
    "prompt": [{"role": "user", "content": "Sort these names: Alice, Bob"}],
    "info": {
        "follow_ups": ["Now add Charlie", "Now add Dan"],
        "ground_truths": [["Alice", "Bob"], ["Alice", "Bob", "Charlie"], ["Alice", "Bob", "Charlie", "Dan"]],
        "num_turns": 3,
    }
}
```

**State** — a dict that carries everything through the rollout. Your dataset row fields (`prompt`, `answer`, `info`, `task`) are accessible via `state["info"]`, `state["answer"]`, etc. Runtime fields get added during the rollout: `trajectory`, `completion`, `reward`, `is_completed`.

**`env_response(messages, state)`** — your function. Called after each model turn. Gets the full conversation so far + state. Returns new messages to append. This is where your environment logic lives.

**`@vf.stop`** — methods that check if the rollout is done. Called after each turn. Return `True` to end.

### The Loop

```
1. Model sees initial prompt → responds
2. Response appended to conversation
3. Check @vf.stop → if done, go to 6
4. env_response() called → returns new messages
5. New messages appended to conversation → go to 1
6. Whole conversation scored by rubric
```

The messages list keeps growing until stop. By the end it's the full conversation:

```
Turn 1: [{user: "Sort: Alice, Bob"}, {assistant: "Alice, Bob"}, {user: "Now add Charlie"}]
Turn 2: [...all above..., {assistant: "Alice, Bob, Charlie"}, {user: "Now add Dan"}]
Turn 3: [...all above..., {assistant: "Alice, Bob, Charlie, Dan"}] → @vf.stop → score
```

### Where Data Comes From

**Pre-planned** — store follow-up questions in `info`, pull them out in `env_response`:

```python
async def env_response(self, messages, state):
    turn = len([m for m in messages if m["role"] == "assistant"]) - 1
    follow_ups = state["info"]["follow_ups"]
    return [{"role": "user", "content": follow_ups[turn]}]
```

**Dynamic** — generate responses based on what the model did:

```python
async def env_response(self, messages, state):
    last_response = messages[-1]["content"]
    result = execute_action(last_response)  # run in browser, sandbox, etc.
    return [{"role": "user", "content": result}]
```

**Hybrid** — dataset has the scaffolding, `env_response` adapts based on what happens. Most real environments are this.

### SingleTurnEnv vs MultiTurnEnv

`SingleTurnEnv` — one prompt, one response, score. No `env_response` needed. Use `question` string or `prompt` messages list.

`MultiTurnEnv` — conversation grows turn by turn. Must override `env_response`. The dataset `prompt` is just the seed.

### final_env_response

If your `env_response` sets `state["final_env_response"]`, the loop stops without making another model call. Used when the environment knows the conversation is done and wants to add a final message (e.g. "Game over, you scored 8/10") without waiting for the model to respond.

### The Environment Hierarchy

Everything is a `MultiTurnEnv` with a different `env_response`:

```
Environment (base)
  └── MultiTurnEnv (adds the env_response loop)
        ├── SingleTurnEnv (one turn, no env_response)
        ├── ToolEnv (env_response = execute tool calls, return results)
        │     ├── StatefulToolEnv (persistent state between turns)
        │     │     ├── SandboxEnv (tools run in isolated sandbox)
        │     │     └── BrowserEnv (browser automation)
        │     └── MCPEnv (MCP server tools)
        ├── OpenEnvEnv (talks to OpenEnv server)
        ├── TextArenaEnv (game environments)
        └── CliAgentEnv (terminal/CLI agent in sandbox)
```

The pattern is always the same inside `env_response`:

1. Parse the model's last message (what action did it take?)
2. Execute that action (click browser, run CLI command, call API, step game)
3. Get back an observation (new page state, command output, game board)
4. Return the observation as messages

For example, OpenEnvEnv's `env_response`:
- Parses the model's action from its text response
- Sends `client.step(action)` to the OpenEnv server
- Gets back `observation`, `reward`, `done`
- Renders the observation into messages and returns them

The model never talks to the real system directly. `env_response` is always the bridge.

### Alphabet-Sort Example

Real multi-turn environment from verifiers. Dataset holds pre-planned follow-ups, `env_response` feeds them one at a time:

```python
class SortingEnv(vf.MultiTurnEnv):
    @vf.stop
    async def max_turns(self, state):
        return len(state["trajectory"]) >= state["info"]["num_turns"]

    async def env_response(self, messages, state):
        turn = len([m for m in messages if m["role"] == "assistant"]) - 1
        return [{"role": "user", "content": state["info"]["follow_ups"][turn]}]
```
