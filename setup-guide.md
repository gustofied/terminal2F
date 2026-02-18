# Clean Setup Guide: Training Qwen3-0.6B with GRPO

Step-by-step instructions that actually work. No shortcuts, no workarounds.

## Prerequisites

- GPU node with 2x A6000 48GB (or similar)
- SSH access with your key
- HuggingFace token (write access)
- WandB API key

## Step 1: SSH into the node

```bash
ssh -i ~/.ssh/primeintellect_ed25519 -p <port> root@<ip>
```

## Step 2: Create workspace

```bash
cd /workspace
mkdir -p verifiers-tutorial
cd verifiers-tutorial
```

## Step 3: Install uv and create venv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --seed
source .venv/bin/activate
```

## Step 4: Upload local files to the node

We have verifiers-rl and alphabet-sort locally — no need to clone repos or use prime's registry on the node.

**From your local machine** (not the node):

```bash
# Upload verifiers-rl package (the trainer, vf-vllm, RLConfig)
scp -P <port> -i ~/.ssh/primeintellect_ed25519 -r \
  /Users/adams/projects/terminal2F/verifiers/packages/verifiers-rl \
  root@<ip>:/workspace/verifiers-tutorial/verifiers-rl

# Upload alphabet-sort environment
scp -P <port> -i ~/.ssh/primeintellect_ed25519 -r \
  /Users/adams/projects/terminal2F/environments/alphabet-sort \
  root@<ip>:/workspace/verifiers-tutorial/alphabet-sort

# Upload the training script
scp -P <port> -i ~/.ssh/primeintellect_ed25519 \
  /Users/adams/projects/terminal2F/training_script.py \
  root@<ip>:/workspace/verifiers-tutorial/training_script.py
```

## Step 5: Install verifiers-rl with ALL dependencies

Back on the node. This is the critical step. Do NOT use `--no-deps`. Do NOT start vLLM first.

```bash
cd /workspace/verifiers-tutorial
uv pip install ./verifiers-rl
```

This installs everything at compatible versions:
- `vllm>=0.10.0,<0.11.0` (pinned — newer versions break vf-vllm)
- `flash-attn` (compiles CUDA kernels from source, ~15 minutes)
- `deepspeed` (required — imported unconditionally at startup)
- `liger-kernel` (required — imported unconditionally in model loading)
- `trl`, `accelerate`, `peft` (training framework)
- `transformers`, `torch` (at compatible versions)

**Wait for flash-attn to finish compiling.** It takes 10-15 minutes and uses significant RAM. This is why vLLM must NOT be running — both compete for memory and the node can OOM and crash (we learned this the hard way).

## Step 6: Install extras

```bash
# WandB for training curve monitoring (not bundled)
uv pip install wandb

# Install alphabet-sort into your venv so Python can import it
uv pip install -e /workspace/verifiers-tutorial/alphabet-sort
```

## Step 7: Create config files

```bash
mkdir -p configs

cat > configs/endpoints.toml <<'EOF'
[[endpoint]]
endpoint_id = "Qwen3-0.6B"
model = "willcb/Qwen3-0.6B"
url = "http://0.0.0.0:8000/v1"
key = "EMPTY"
EOF

cat > configs/eval.toml <<'EOF'
[[eval]]
env_id = "alphabet-sort"
model = "Qwen3-0.6B"
num_examples = 5
rollouts_per_example = 3
max_tokens = 1024
save_results = true
save_to_hf_hub = true
hf_hub_dataset_name = "gustofied/Qwen3-0.6B-alphabet-sort-eval"
EOF
```

## Step 8: Training script

The training script was already uploaded in Step 4 (`training_script.py`). It lives at `/workspace/verifiers-tutorial/training_script.py` on the node.

See `training_script.py` locally for the source. Key differences from the blog tutorial:
- Import directly from `verifiers_rl`, not through `vf.*` wrappers
- Use `RLTrainer` not `GRPOTrainer` (the wrapper has a broken arg order)
- Use `RLConfig` not `grpo_defaults()` (identical, just not deprecated)
- Use the actual field names: `micro_batch_size`, `rollouts_per_example`, `batch_size`
- Explicitly set `use_lora=False` (defaults to True, tutorial wants full fine-tuning)
- Use keyword args for RLTrainer to avoid positional arg confusion

## Step 9: Start vf-vllm on GPU 0

```bash
export HF_TOKEN=your_hf_token_here
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests
```

Wait for `Application startup complete.` before proceeding.

**Why vf-vllm and NOT vllm serve?** Training needs weight sync. After each training step, the updated model weights are pushed from GPU 1 to GPU 0 via NCCL. `vf-vllm` adds custom HTTP endpoints (`/init_communicator`, `/update_weights`, `/get_world_size`) that enable this. Regular `vllm serve` doesn't have these — you'll get a 404 on `/get_world_size`.

**Why willcb/Qwen3-0.6B?** The original `Qwen/Qwen3-0.6B` strips `<think>` sections from context between turns, breaking multi-turn GRPO training. The `willcb` fork has a modified chat template that preserves them.

## Step 10: Run baseline eval (optional)

Open a second SSH terminal:

```bash
cd /workspace/verifiers-tutorial && source .venv/bin/activate
export HF_TOKEN=your_hf_token_here
vf-eval configs/eval.toml -e configs/endpoints.toml
```

Expected baseline: ~0.4 mean reward.

## Step 11: Run training on GPU 1

In the second SSH terminal:

```bash
export HF_TOKEN=your_hf_token_here
export WANDB_API_KEY=your_wandb_key_here
CUDA_VISIBLE_DEVICES=1 python training_script.py
```

Training takes ~8 hours for 1000 steps. Monitor at wandb.ai.

## Step 12: Evaluate the trained model

After training completes, kill vf-vllm (Ctrl+C) and serve the trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 vf-vllm --model ./mymodel --enforce-eager --disable-log-requests
```

Update endpoints and run eval:

```bash
cat > configs/endpoints-tuned.toml <<'EOF'
[[endpoint]]
endpoint_id = "Qwen3-0.6B-tuned"
model = "./mymodel"
url = "http://0.0.0.0:8000/v1"
key = "EMPTY"
EOF

cat > configs/eval-tuned.toml <<'EOF'
[[eval]]
env_id = "alphabet-sort"
model = "Qwen3-0.6B-tuned"
num_examples = 5
rollouts_per_example = 3
max_tokens = 1024
save_results = true
save_to_hf_hub = true
hf_hub_dataset_name = "gustofied/Qwen3-0.6B-tuned-alphabet-sort-eval"
EOF

vf-eval configs/eval-tuned.toml -e configs/endpoints-tuned.toml
```

Expected after training: ~0.58+ mean reward (~43% improvement).

## What Can Go Wrong

| Problem | Cause | Fix |
|---------|-------|-----|
| flash-attn OOM during install | vLLM running at the same time | Kill vLLM, install first, start vLLM after |
| `vf-vllm: command not found` | Not in venv | `source .venv/bin/activate` |
| `No module named 'alphabet_sort'` | Environment not in project venv | `uv pip install alphabet-sort==0.1.5 --extra-index-url ...` |
| 404 on `/get_world_size` | Used `vllm serve` instead of `vf-vllm` | Training requires `vf-vllm` for weight sync |
| NCCL error after Ctrl+C | Stale communicator from previous run | Restart BOTH vf-vllm and training |
| `401 Unauthorized` on HF Hub | Missing `HF_TOKEN` | `export HF_TOKEN=xxx` |
| wandb prompts interactively | Missing `WANDB_API_KEY` | `export WANDB_API_KEY=xxx` |
| SSH disconnects, training dies | Long-running process in foreground | Use `tmux` or `screen` to persist sessions |

## Pro Tips

- **Use tmux** so training survives SSH disconnects: `tmux new -s train`, then Ctrl+B D to detach, `tmux attach -t train` to reconnect
- **Revoke HF tokens** if you accidentally paste them in chat
- **Check GPU usage** from another terminal: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
- **scp uses uppercase -P for port**: `scp -P <port> -i ~/.ssh/key file root@ip:/path/`
