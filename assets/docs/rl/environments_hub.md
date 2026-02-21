## Environments Hub and Verifiers

### What It Is

A community platform for sharing RL training environments and evaluation tasks. Think PyPI but for environments — each one is a versioned Python package with a `pyproject.toml`, distributed as a wheel.

Browse environments at the [Environments Hub](https://hub.primeintellect.ai/environments). Install with `prime env install owner/env-name@version`.

### Why It Exists

The current ecosystem has a few problems:

**No shared platform for RL environments.** Interest in training LLMs with RL is growing fast, but there's no community place to explore and share train-ready environments. Everyone builds their own in isolation.

**Environments are locked to specific trainers.** Most environment implementations are tied to one RL stack and hard to adapt to a different trainer. If you switch from prime-rl to your own trainer, you often can't reuse the environments without rewriting.

**Evals and RL environments are the same thing but treated separately.** Both are just dataset + harness + scoring rules. But eval suites (lm_eval, lighteval, HELM) and RL environments live in completely different ecosystems with no shared spec. You end up implementing the same task twice — once for eval, once for training.

**Eval suites don't handle agentic tasks well.** Popular eval suites work great for single-turn Q&A, but tasks that are agentic or need complex infrastructure (TAU-bench, TerminalBench, SWE-bench) end up as independent repos without shared entrypoints.

**Monorepos don't scale for environments.** Realistic agent environments have their own dependencies and versioning needs. Stuffing them all into one repo gets unmaintainable fast.

### How It Works

Environments follow the verifiers spec. They declare dependencies in `pyproject.toml` and are distributed as wheels. Development focuses on the task-specific parts — datasets, tools, reward functions — and automatically gets the infrastructure for both evaluation and RL training.

### Creating an Environment

```bash
# authenticate with the hub
prime login

# scaffold a new environment
prime env init my-env --path ./environments/

# install it as editable so changes take effect immediately
uv add --editable ./environments/my_env
```

This gives you a folder with `pyproject.toml`, a module file, and a README. The editable install means you can hack on the code and test without reinstalling.

### Testing Locally

Both work with locally installed environments — no need to push to hub first:

```bash
# vf-eval — verifiers' built-in eval runner
uv run vf-eval my-env

# prime eval — looks for locally installed packages by short name
prime eval run my-env -m model-name
```

If you use the full slug (`prime eval run owner/my-env`), it auto-installs from the hub. Short name just looks at what's in your local venv.

### Upload Your Environment

Once you've developed and tested your environment, push it to the Environments Hub:

```bash
# inside the environments/my_env/ directory
prime env push
```

Others can then install it with `prime env install your-name/my-env@0.1.0`.

### The Stack

- **[Prime CLI](https://github.com/PrimeIntellect-ai/prime-cli)** — install, upload, manage environments
- **[Verifiers](https://github.com/PrimeIntellect-ai/verifiers)** — modular components for creating environments and training agents
- **[Prime RL](https://github.com/PrimeIntellect-ai/prime-rl)** — large-scale RL training with FSDP
