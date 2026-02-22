## Verifiers - Small Learnings

Things worth knowing when working with the verifiers library.

### Dataset auto-wrapping

If your dataset rows have a `question` string column, Verifiers will auto-wrap it into a user message for you:

```
# you provide:
"question": "What is 2+2?"

# it becomes:
prompt = [{"role": "user", "content": "What is 2+2?"}]
```

You don't need to manually build the messages list. Just give it a string and it handles the chat format.

### Environment references

Three formats depending on what you're doing:

- **Hub reference** — `owner/name@version` — for installing from the hub:
  ```bash
  prime env install owner/my-env@0.1.0
  ```

- **Hub slug** — `owner/name` — for running eval via prime (auto-installs if not local):
  ```bash
  prime eval run owner/my-env -m gpt-4.1-mini -n 10 -r 3
  ```

- **Environment ID** — just the name — for local eval with vf-eval:
  ```bash
  uv run vf-eval my-env
  ```

Under the hood it's all the same: the environment ID gets converted to a Python module name (hyphens → underscores) and imported. `prime eval` can also resolve from the hub by slug, `vf-eval` only looks locally.

### Rubric vs reward function

A rubric is a collection of reward functions. Each reward function scores one aspect (e.g. `to_correct`, `cc_correct`, `bcc_correct`), and the rubric combines them into the final reward score.

In verifiers, you pass multiple scoring functions with weights to the rubric:

```python
rubric = vf.Rubric(
    funcs=[to_correct, cc_correct],
    weights=[1.0, 1.0],
)
```

### Reward function parameters

Verifiers auto-maps parameter names to available data. You can ask for any combination:

```python
def score(completion, answer, **kwargs)   # completion + answer column
def score(completion, info, **kwargs)     # completion + info dict
def score(completion, state, **kwargs)    # completion + full state
def score(state, **kwargs)               # just state, pull everything yourself
```

`async` is optional — verifiers uses `maybe_await` so both sync and async reward functions work. Use `async` when your reward function makes API calls (e.g. LLM-as-judge).

### `info` is static, `state` is live

`info` is a dict on the dataset row — it stays the same for the entire rollout. Put ground truths, follow-up questions, rules, or any reference data here.

`state` updates every turn — it holds the growing trajectory, completion, turn count, rewards, etc.

For multi-turn, index into `info` by turn number:

```python
turn = len(state["trajectory"])
expected = state["info"]["expected"][turn]
```

### Configuration knobs go on `load_environment`

`--env-args` from the CLI get passed as kwargs to `load_environment()`. So any knob you want configurable from outside goes on that function signature with a default:

```python
def load_environment(max_turns: int = 1, difficulty: str = "easy", **kwargs):
    dataset = build_dataset(difficulty)
    return MyEnv(dataset=dataset, rubric=rubric, max_turns=max_turns)
```

Then from the CLI:

```bash
prime eval run my-env --env-args '{"max_turns": 3, "difficulty": "hard"}'
```

`load_environment` receives it, passes it down to the class or dataset builder. The class itself doesn't need to know about CLI args — it just takes constructor params like any normal class.

### `answer` is just a convention

Nothing in verifiers requires an `answer` column. It's just a dataset column name that gets passed to reward functions. You can use `info` for everything, or any column name you want — as long as your reward function parameters match.
