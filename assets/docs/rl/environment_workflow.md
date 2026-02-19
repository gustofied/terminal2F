## Environment Workflow

Environments are versioned Python packages - importable, reproducible, pinned with deps.

### The Two Modes

**Consumer mode** - install and use, can't edit:
```bash
prime env install owner/env-name
```
Goes into site-packages. Meant for running, not editing.

**Developer mode** - pull source, install editable, hack on it:
```bash
prime env pull owner/env-name@latest --target ./environments/env-name
uv add --editable ./environments/env-name
```
Source is in your tree. Changes take effect immediately.

### Preferred Workflow

Always pull + editable. Never blind install.

Why: you always want to see what you're running. Editable installs mean you can read and modify the environment code directly, debug reward functions, tweak prompts - without reinstalling anything. When something breaks during training you can trace it right to the source. Blind installs hide the code in site-packages where you'll never look at it.

```bash
# 1. Pull the source
prime env pull owner/env-name@latest --target ./environments/env-name

# 2. Make it editable in your project (recorded in pyproject.toml)
uv add --editable ./environments/env-name

# 3. Hack on it, run eval, iterate
vf-eval configs/eval.toml -e configs/endpoints.toml

# 4. Push changes back (auto-bumps version)
prime env push --auto-bump
```

`--auto-bump` automatically increments the version number in pyproject.toml so you don't need to manually update it each time you push.

### Quick vs Recorded Install

| Command | What it does | Recorded in pyproject.toml? |
|---|---|---|
| `uv pip install -e ./path` | Editable install into venv | No - `uv sync` can forget it |
| `uv add --editable ./path` | Editable install + writes path dependency | Yes - stays part of project |

Use `uv add --editable` when you want it to persist. Use `uv pip install -e` for quick throwaway testing.

### Creating a New Environment

```bash
# Scaffold files (does NOT make it importable)
prime env init my-env --path ./environments/my-env

# Make it importable
uv add --editable ./environments/my-env
```

`prime env init` only creates files. You still need the editable install for Python to find it.

### Notes

- `vf-eval` defaults to Prime Inference unless you configure an endpoint or pass `-b` and `-k`. Preferably configure endpoints via a TOML file (`configs/endpoints.toml`) so you don't have to pass flags every time - keeps commands short and settings reproducible.
- For remote GPU nodes: scp your environment folder, then `uv pip install -e ./path` on the node. Same editable workflow - you can still read and tweak the code on the node.
- `--target` is for `prime env pull`, `--path` is for `prime env init`
