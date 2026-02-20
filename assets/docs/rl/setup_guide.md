## Setup Guide: GRPO Training on a GPU Node

Based on the [PrimeIntellect tutorial](https://www.primeintellect.ai/blog/verifiers-tutorial), adjusted for our local setup.

Step-by-step for training Qwen3-0.6B on alphabet-sort using verifiers-rl.

### Prerequisites

- GPU node with 2x GPUs (e.g. 2x A6000 48GB, 2x RTX Pro 6000 96GB)
- SSH access with your key (see GPU Nodes doc)
- HuggingFace token (write access for push_to_hub)
- WandB API key

### Step 1: SSH in

```bash
ssh -i ~/.ssh/primeintellect_ed25519 -p <port> root@<ip>
```

### Step 2: Create workspace

```bash
cd /workspace
mkdir -p verifiers-tutorial && cd verifiers-tutorial
```

### Step 3: Install uv and create venv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Why `--seed`? It installs pip and setuptools into the venv. Not strictly needed with uv, but some packages (like flash-attn) shell out to pip during build. Without `--seed` those builds can fail.

### Step 4: Upload local files

From your local machine (not the node):

```bash
# verifiers-rl package
scp -P <port> -i ~/.ssh/primeintellect_ed25519 -r \
  ~/projects/terminal2F/external/verifiers/packages/verifiers-rl \
  root@<ip>:/workspace/verifiers-tutorial/verifiers-rl

# alphabet-sort environment
scp -P <port> -i ~/.ssh/primeintellect_ed25519 -r \
  ~/projects/terminal2F/environments/alphabet-sort \
  root@<ip>:/workspace/verifiers-tutorial/alphabet-sort

# training script
scp -P <port> -i ~/.ssh/primeintellect_ed25519 \
  ~/projects/terminal2F/training_script.py \
  root@<ip>:/workspace/verifiers-tutorial/training_script.py
```

**Why scp instead of cloning or pip install?**

In a normal setup you'd just `pip install verifiers-rl` from PyPI and `prime env install owner/alphabet-sort` from the hub. But:

- **verifiers-rl** wasn't published as a standalone package on PyPI — it lives inside the verifiers monorepo under `packages/verifiers-rl/`. You can't `pip install` it directly. So we scp the directory and install from the local copy.
- **alphabet-sort** was a local/private environment we were developing, not yet on the hub. Same deal — scp it over, install editable.
- **training_script.py** was our custom Python script with corrected RLConfig fields (see Training Script doc). Now this could be replaced with a TOML config and `vf-rl @ config.toml` instead.

If everything were published (verifiers-rl on PyPI, environment on the hub), you'd skip scp entirely and just install on the node. The scp approach is for when you're working with local/unpublished code.

### Step 5: Install verifiers-rl (with all deps)

Back on the node. This is the critical step - do NOT use `--no-deps`.

```bash
cd /workspace/verifiers-tutorial
uv pip install ./verifiers-rl
```

This installs everything at compatible versions: vllm, flash-attn (compiles from source, ~15 min), deepspeed, liger-kernel, trl, accelerate, torch.

**Do NOT start vLLM during this step.** flash-attn compilation eats RAM. If vLLM is running too, the node can OOM and crash.

If flash-attn OOM's during compile even without vLLM: `MAX_JOBS=4 uv pip install ./verifiers-rl`

### Step 6: Install extras

```bash
uv pip install wandb
uv pip install -e ./alphabet-sort
```

### Step 7: Start vf-vllm on GPU 0

```bash
export HF_TOKEN=your_token
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B \
  --enforce-eager --disable-log-requests
```

Wait for `Application startup complete.`

Why `vf-vllm` not `vllm serve`? Training needs weight sync. After each step the trainer pushes updated weights from GPU 1 to GPU 0 via NCCL. `vf-vllm` adds `/init_communicator`, `/update_weights`, `/get_world_size` endpoints for this. Regular `vllm serve` gives 404 on those.

Why `willcb/Qwen3-0.6B`? The original Qwen3 strips `<think>` sections between turns, breaking multi-turn GRPO. The willcb fork preserves them.

### Step 8: Run training on GPU 1

In a second tmux pane:

```bash
cd /workspace/verifiers-tutorial && source .venv/bin/activate
export HF_TOKEN=your_token
export WANDB_API_KEY=your_key
CUDA_VISIBLE_DEVICES=1 python training_script.py
```

Monitor at wandb.ai. Training takes ~8 hours for 1000 steps.

There's also a TOML-based approach (`vf-rl @ config.toml`) that automates the two-pane tmux setup. See Training Script doc. Either way works.

### Step 9: Evaluate

Kill vf-vllm, serve the trained model, run eval:

```bash
CUDA_VISIBLE_DEVICES=0 vf-vllm --model ./mymodel \
  --enforce-eager --disable-log-requests
```

### Common Errors

| Problem | Fix |
|---------|-----|
| flash-attn OOM during install | Kill vLLM first, or `MAX_JOBS=4` |
| `vf-vllm: command not found` | `source .venv/bin/activate` |
| 404 on `/get_world_size` | Used `vllm serve` instead of `vf-vllm` |
| `401 Unauthorized` on HF Hub | `export HF_TOKEN=xxx` (needs write access) |
| NCCL error after Ctrl+C | Restart both vf-vllm and training |
| SSH disconnect kills training | Use tmux |
| CUDA OOM on backward pass | Reduce `micro_batch_size` (4 instead of 8) |

### Tips

- Use tmux: `tmux new -s train`, Ctrl+B D to detach, `tmux attach -t train` to reconnect
- Split tmux: Ctrl+B % (vertical), Ctrl+B " (horizontal), Ctrl+B arrow to switch
- Check GPU usage: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
- scp uses uppercase `-P` for port
