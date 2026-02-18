# Learnings: RL Training on Rented GPUs

Everything we learned the hard way, organized for future reference.

## GPU Node Setup

### SSH
```bash
ssh -i ~/.ssh/primeintellect_ed25519 -p <port> root@<ip>
```
- scp uses uppercase `-P` for port: `scp -P <port> -i ~/.ssh/key file root@ip:/path/`
- Always use tmux so long-running processes survive SSH disconnects

### tmux Cheat Sheet
```
tmux new -s <name>          # create named session
tmux attach -t <name>       # reattach to session
Ctrl+B d                    # detach (leave running)
Ctrl+B c                    # new window
Ctrl+B n / Ctrl+B p         # next/previous window
Ctrl+B "                    # split horizontal
Ctrl+B %                    # split vertical
Ctrl+B arrow                # switch pane
```

### Environment Setup
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Monitoring
```bash
nvidia-smi                                                          # quick GPU check
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv  # compact
watch -n 2 nvidia-smi                                               # live monitor
free -h                                                             # RAM check
```

## Package Installation

### flash-attn is always painful
- Compiles CUDA kernels from source (~15 min)
- Needs torch installed first (undeclared build dependency)
- Uses massive RAM per nvcc process — limit with `MAX_JOBS=4` (or lower if OOM)
- Must install BEFORE starting vLLM — both compete for memory
- The full incantation:
  ```bash
  uv pip install torch
  uv pip install hatchling
  MAX_JOBS=4 uv pip install <package> --no-build-isolation
  ```

### Build isolation (`--no-build-isolation`)
- Normally pip/uv builds packages in an isolated temp venv
- `--no-build-isolation` builds against your current environment
- Needed when a package requires torch to compile but doesn't declare it
- Tradeoff: ALL packages lose isolation, so build tools (hatchling etc.) must be manually installed

### Never use `--no-deps`
- Skips ALL dependency resolution — you lose version pins
- Creates cascading version mismatches discovered one at a time through runtime crashes
- Always worse than waiting for a long compile
- If flash-attn OOMs, use `MAX_JOBS=2` instead of `--no-deps`

### transformers version conflicts
- vllm 0.10.x needs `transformers<5` — the `all_special_tokens_extended` attribute was removed in transformers 5.x
- If you see `has no attribute all_special_tokens_extended`: `uv pip install "transformers<5"`

## The Verifiers Ecosystem

### What's what
| Package | What it does | Where to get it |
|---------|-------------|-----------------|
| `verifiers` | Environments + eval (`vf-eval`) | PyPI |
| `verifiers-rl` | Legacy trainer + `vf-vllm` | Git clone from `PrimeIntellect-ai/verifiers`, under `packages/verifiers-rl/` |
| `prime-rl` | Recommended production trainer | Separate repo |
| `vllm` | Model serving | PyPI |

### verifiers-rl is a monorepo subdirectory
```
PrimeIntellect-ai/verifiers/
├── packages/
│   └── verifiers-rl/    ← trainer, vf-vllm, RLConfig (NOT on PyPI)
├── src/
│   └── verifiers/       ← environments, eval (this IS on PyPI)
```

### verifiers-rl pins old versions
- `vllm>=0.10.0,<0.11.0` (July 2025 — vLLM is at 0.16+ now)
- `flash-attn==2.8.3`
- Abandoned in November 2025. Will eventually stop working.

### vf-vllm vs vllm serve
- `vf-vllm`: adds `/get_world_size`, `/init_communicator`, `/update_weights` endpoints for NCCL weight sync. **Required for training.**
- `vllm serve`: standard serving. Works for eval only.
- If you see 404 on `/get_world_size` — you used `vllm serve` instead of `vf-vllm`

### willcb/Qwen3-0.6B vs Qwen/Qwen3-0.6B
- Original Qwen3 strips `<think>` tokens from context between turns
- This breaks multi-turn GRPO training (the model can't see its own reasoning from previous turns)
- `willcb` fork preserves `<think>` in the chat template

## GRPO Training

### How GRPO works
1. Sample a batch of prompts
2. For each prompt, generate N completions (rollouts_per_example)
3. Score each completion with the environment's reward function
4. Use the group mean as the baseline (no need for a separate value model like PPO)
5. Completions above the mean get positive advantage, below get negative
6. Update the policy to increase probability of high-advantage completions

### RLConfig field names (NOT what the tutorial says)
| Tutorial says | Actual RLConfig field | What it means |
|--------------|----------------------|---------------|
| `per_device_train_batch_size` | `micro_batch_size` | Per GPU per step |
| `num_generations` | `rollouts_per_example` | GRPO group size |
| `gradient_accumulation_steps` | (hardcoded to 1) | Can't change |
| N/A | `batch_size` | Total rollouts per batch |

### GRPOTrainer wrapper is broken
- `vf.GRPOTrainer` passes `(model, processing_class, env, args)` but `RLTrainer` expects `(model, env, args, processing_class)`
- Always use `RLTrainer` directly with keyword args

### Key training parameters
```python
args = RLConfig(
    use_liger=True,          # needs flash-attn
    use_lora=False,          # False = full fine-tuning (default is True!)
    batch_size=64,           # total rollouts per batch
    micro_batch_size=4,      # per GPU — reduce if OOM
    rollouts_per_example=8,  # GRPO group size
    max_seq_len=2048,
    max_steps=1000,
)
```

### OOM during training
- Reduce `micro_batch_size` first (8 → 4 → 2)
- Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to help with fragmentation
- Reduce `max_seq_len` if still OOMing
- Reduce `batch_size` as last resort (changes training dynamics)

## WandB Charts — What to Watch

| Chart | What it tells you | Good sign |
|-------|------------------|-----------|
| `train/reward` | Main metric — is the model learning? | Trending up |
| `train/loss` | Policy loss | Bouncing around 0-0.05 is normal |
| `train/entropy` | How exploratory the model is | Stays above 0.3 (collapse = bad) |
| `train/mismatch_kl` | Drift between training and inference model | Near 0 = weight sync working |
| `train/advantage/absmean` | GRPO signal strength | Some signal > 0 means learning |
| `train/grad_norm` | Gradient magnitudes | Stable, no explosions |

## Checkpoints

- `save_steps=100` saves every 100 steps
- `save_total_limit=1` keeps only the latest
- `push_to_hub=True` pushes to HF Hub at each save
- Ctrl+C loses progress since last checkpoint only
- Evaluate any checkpoint by serving it: `vf-vllm --model ./mymodel`

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| flash-attn `No module named 'torch'` | Undeclared build dep | `uv pip install torch` first, then `--no-build-isolation` |
| flash-attn `Killed` during compile | OOM on CPU RAM | `MAX_JOBS=2` or `MAX_JOBS=4` |
| `No module named 'hatchling'` | `--no-build-isolation` disables all isolation | `uv pip install hatchling` |
| `all_special_tokens_extended` | transformers 5.x breaks vllm 0.10 | `uv pip install "transformers<5"` |
| `401 Unauthorized` on HF Hub | Token missing or read-only | Need write-access token, `export HF_TOKEN=xxx` |
| `CUDA out of memory` during backward | Batch too large | Reduce `micro_batch_size` |
| `vf-vllm: command not found` | Not in venv, or verifiers-rl not installed | `source .venv/bin/activate` |
| 404 on `/get_world_size` | Used `vllm serve` instead of `vf-vllm` | Training needs `vf-vllm` for weight sync |
| NCCL errors | Stale communicator or version mismatch | Restart BOTH vf-vllm and training |
| wandb prompts interactively | Missing key | `export WANDB_API_KEY=xxx` |

## Workflow Template

For any future training run on a rented GPU node:

```bash
# 1. SSH in, set up tmux
ssh -i ~/.ssh/key -p <port> root@<ip>
tmux new -s work

# 2. Create workspace + venv
cd /workspace && mkdir -p project && cd project
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --seed && source .venv/bin/activate

# 3. scp files from local machine (from another terminal)
scp -P <port> -i ~/.ssh/key -r local/files root@<ip>:/workspace/project/

# 4. Install dependencies (flash-attn first, vLLM after)
uv pip install torch hatchling
MAX_JOBS=4 uv pip install ./your-package --no-build-isolation

# 5. Set tokens
export HF_TOKEN=xxx
export WANDB_API_KEY=xxx

# 6. Start inference server (tmux window 0)
CUDA_VISIBLE_DEVICES=0 vf-vllm --model <model> --enforce-eager --disable-log-requests

# 7. Start training (tmux window 1: Ctrl+B c)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python train.py

# 8. Detach and walk away: Ctrl+B d
# 9. Reattach later: tmux attach -t work
```

## Future Goal: Extract verifiers-rl

See `TODO-verifiers-rl-extraction.md`. The plan:
- Extract multi-turn orchestrator from verifiers-rl
- Port to TRL's `rollout_func` parameter
- Drop old version pins (vllm, flash-attn)
- Make a standalone, maintainable module
