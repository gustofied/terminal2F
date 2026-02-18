# The Verifiers Ecosystem: A Moving Target

How the verifiers library evolved, what broke, and what the current state of training actually looks like.

## The Origin Story

**William Brown** (willccbb on GitHub, willcb on PyPI) built verifiers as an independent, self-contained library in early 2025. The idea was clean: one package that gives you environments (tasks for LLMs to learn), evaluation tooling, and a built-in GRPO trainer. Install one thing, train a model. It was small, opinionated, and worked.

The original API looked like this:

```python
import verifiers as vf

model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen3-0.6B")
env = vf.load_environment("alphabet-sort")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=env, args=vf.grpo_defaults())
trainer.train()
```

Everything in one import. The trainer knew how to talk to vLLM, run multi-turn conversations, score rollouts, compute GRPO advantages, and update weights. For small-scale experiments (single node, 1-2 GPUs), it was the right level of abstraction.

## Prime Intellect Takes Over

Mid-2025, Prime Intellect adopted verifiers into their ecosystem. Will joined (or partnered with) Prime. The repo moved from `willccbb/verifiers` to `PrimeIntellect-ai/verifiers`. Will stayed as the maintainer.

This made sense. Prime Intellect was building an "open superintelligence stack" — they needed standardized environments for their large-scale training runs (INTELLECT-2, INTELLECT-3). Verifiers' environment format was exactly what they wanted.

But Prime Intellect also had their own training framework: **prime-rl**. A distributed, async RL trainer built for multi-node, multi-GPU scale. It handles things Will's lightweight trainer never needed to — MoE models, distributed weight broadcasting, untrusted inference worker verification, etc.

Two trainers doing the same job. One had to go.

## The Split (v0.1.7, November 2025)

In version 0.1.7, the verifiers library was reorganized:

- **`verifiers`** (on PyPI) became environments + evaluation only. The core value proposition — defining tasks, scoring outputs, running `vf-eval`. This is the part everyone uses and it's well-maintained.

- **`verifiers-rl`** was carved out as a separate package inside the monorepo (`packages/verifiers-rl/`). It contains the legacy trainer (`vf.GRPOTrainer` renamed to `vf.RLTrainer`), `vf.get_model_and_tokenizer`, and `vf-vllm`. It was never published to PyPI.

- **`prime-rl`** became the recommended training path. TOML-configured, handles vLLM orchestration, designed for their infrastructure.

The `verifiers` package on PyPI now throws an error if you try to use any training function:

```
AttributeError: To use verifiers.get_model_and_tokenizer, install as `verifiers-rl`.
```

It tells you to install a package that doesn't exist on PyPI.

## What This Means in Practice

### For Evaluation: Things Got Better

The move to TOML config files for evaluation is genuinely an improvement:

**Before (CLI args + Python endpoints):**
```bash
vf-eval alphabet-sort -m Qwen3-0.6B -e "endpoints.py" -n 5 -r 3 -t 1024 \
  --save-dataset --save-to-hf-hub --hf-hub-dataset-name "user/dataset"
```

**After (TOML configs):**
```bash
vf-eval configs/eval.toml -e configs/endpoints.toml
```

Declarative, reproducible, version-controllable. You can define multiple evaluations in one file. The endpoints config separates "where are my models" from "what do I want to evaluate." This is a good change.

### For Training: Things Got Worse (Then Allegedly Better)

The migration path from the old trainer to the new one is broken:

1. **Every tutorial references `vf.GRPOTrainer`** — including the one on Hugging Face's blog by anakin87, which is the most prominent guide for getting started. It walks you straight into a dead end.

2. **`verifiers-rl` is not on PyPI** — the error message says "install as verifiers-rl" as if it's a pip install. It's not. You have to clone the entire verifiers repo from GitHub and install from a subdirectory:
   ```bash
   git clone --depth 1 https://github.com/PrimeIntellect-ai/verifiers.git /tmp/vf-repo
   uv pip install /tmp/vf-repo/packages/verifiers-rl --no-deps
   ```

3. **`verifiers-rl` depends on flash-attn** which requires compiling CUDA kernels from source. This takes 10-15 minutes, eats all your RAM, and can crash your GPU node if vLLM is also running. On our 2x A6000 node (100GB RAM), the flash-attn build caused an OOM that killed SSH and required a full instance restart.

4. **The "solution" is prime-rl** — which is Prime Intellect's proprietary trainer. It works with TOML configs (`uv run prime-rl configs/train.toml`) and handles orchestration automatically. But it's their tool, tied to their ecosystem. The `prime lab setup` command scaffolds their whole workflow.

5. **Nobody can just swap trainers for multi-turn environments** — TRL's `GRPOTrainer` (HuggingFace's maintained alternative) is single-turn. It generates one completion, scores it, done. Multi-turn environments like alphabet-sort need a conversation loop — the model responds, the environment gives new input, the model responds again. That loop lived in `verifiers-rl`. Without it, you'd have to reimplement it yourself.

## The Current Landscape

| Component | Package | Status | On PyPI? |
|-----------|---------|--------|----------|
| Environments | `verifiers` | Active, maintained | Yes |
| Evaluation (`vf-eval`) | `verifiers` | Active, maintained | Yes |
| Environment Hub | `prime` CLI | Active | Yes (uv tool) |
| Legacy trainer | `verifiers-rl` | Abandoned, not maintained | No |
| Recommended trainer | `prime-rl` | Active, maintained | Separate repo |
| vLLM wrapper (`vf-vllm`) | `verifiers-rl` | Abandoned | No |
| vLLM serving | `vllm` | Active (use `vllm serve` directly) | Yes |

## What Trainer Options Actually Exist?

### prime-rl (Recommended by Prime Intellect)
- TOML config, no Python script needed
- Handles vLLM, training, orchestration in a tmux session
- Built for scale (multi-node, MoE, distributed)
- Tied to Prime Intellect's ecosystem
- Works with verifiers environments natively

### verifiers-rl / vf.GRPOTrainer (Legacy)
- Python script, the approach every tutorial shows
- Simple, single-node, good for learning
- Not on PyPI, painful to install, not maintained
- Depends on flash-attn (compilation nightmare)
- Will eventually stop working as verifiers evolves

### TRL GRPOTrainer (HuggingFace)
- On PyPI, well-maintained, `pip install trl`
- Has vLLM integration (server mode and colocate mode)
- Works great for single-turn tasks (math, coding, formatting)
- Does NOT handle multi-turn conversation loops natively
- Would need a custom `rollout_func` for multi-turn environments

### SkyRL, OpenRLHF, etc.
- Independent RL training frameworks
- Each has their own API and config format
- Would need custom integration with verifiers environments
- Python scripts, not TOML

## The Trainer-Agnostic Promise

The verifiers docs say "verifiers is intended to be largely trainer-agnostic." The environments and reward functions genuinely are portable. You can load an environment with `vf.load_environment("alphabet-sort")` and get its reward function regardless of which trainer you use.

But there's a gap: the **multi-turn conversation loop**. Verifiers environments define multi-turn interactions (the model responds, the environment gives new input, repeat). Someone has to orchestrate that loop. The old `vf.GRPOTrainer` did it. `prime-rl` does it. Other trainers don't — they'd need a wrapper.

For single-turn environments, the trainer-agnostic promise holds. For multi-turn (which is the interesting part — agents, tool use, games), you're pushed toward prime-rl unless you want to write the orchestration yourself.

## Practical Advice

If you're **learning RL for LLMs**:
- Use verifiers for environments and evaluation (the good part)
- Try to get `verifiers-rl` installed from git for the training script (the painful part)
- Install everything before starting vLLM to avoid OOM
- Or bite the bullet and use `prime-rl` with a TOML config

If you're **doing production training**:
- Use `prime-rl` or bring your own trainer
- Don't depend on `verifiers-rl` — it's legacy and will break

If you're **building a new environment**:
- The environment format (`MultiTurnEnv`, `@vf.stop`, `env_response`) is stable and well-designed
- Your environment will work with any trainer that integrates with verifiers
- Focus on the reward function — that's the portable part

## Timeline

| Date | Event |
|------|-------|
| Jan 2025 | Will Brown creates verifiers (v0.0.0). Self-contained: environments + eval + trainer |
| Mid 2025 | Prime Intellect adopts verifiers. Repo moves to PrimeIntellect-ai org |
| Jun 2025 | v0.1.0 — first stable release |
| Nov 2025 | v0.1.7 — trainer split into `verifiers-rl`, `vf.RLTrainer` replaces `vf.GRPOTrainer`, prime-rl becomes recommended |
| Nov 2025 | v0.1.8 — major rollout system refactor |
| Jan 2026 | v0.1.9 — experimental environment classes |
| Feb 2026 | v0.1.10 — current stable. Training functions throw "install verifiers-rl" error |
