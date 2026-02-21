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

In verifiers, a rubric has multiple scoring functions decorated with weights, and it aggregates them:

```python
@rubric.metric(weight=1.0)
def to_correct(...): ...

@rubric.metric(weight=1.0)
def cc_correct(...): ...
```
