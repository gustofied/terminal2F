# Training a Small Language Model with Reinforcement Learning

A hands-on guide to evaluating and training Qwen3-0.6B on the alphabet-sort task using the Verifiers ecosystem.

## What Are We Actually Doing?

We're teaching a small language model (600M parameters) to sort names alphabetically across multiple conversation turns. The model starts with a ~40% success rate and, through reinforcement learning (GRPO), learns to do it better.

This is a microcosm of how larger models are trained to follow instructions, use tools, and reason — the same loop, just on a toy task where you can see results in under an hour.

## The Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Environment | `alphabet-sort` | Defines the task, generates prompts, scores outputs |
| Inference | `vllm serve` | Serves the model as an OpenAI-compatible API |
| Evaluation | `vf-eval` | Runs the model against the environment, collects scores |
| Training | `vf.GRPOTrainer` or `prime-rl` | Updates model weights using GRPO |
| Package manager | `uv` | Manages Python environments and dependencies |
| Environment hub | `prime` CLI | Installs versioned environments from Prime Intellect |
| Monitoring | `wandb` | Live training curves |
| Model hosting | Hugging Face Hub | Save checkpoints and eval datasets |

## The Alphabet-Sort Task

The model receives a list of names and must sort them alphabetically by first name, wrapped in XML tags:

```
Input:  Sort these names: MarcoEllero, MassimoTessarotto, EnricoFonda

Output: <alphabetical_sorted>
        EnricoFonda
        MarcoEllero
        MassimoTessarotto
        </alphabetical_sorted>
```

It's multi-turn — the model gets several rounds of names to sort and must maintain context across turns. The reward function uses `difflib` sequence similarity raised to the 4th power, which harshly penalizes small mistakes (getting 90% right only scores ~65%).

## Part 1: Setting Up the GPU Node

### Why a GPU node?

Even a 0.6B model needs GPU memory for inference, and training needs a second GPU. We rent 2x A6000 (48GB each) from Prime Intellect — one for serving, one for training. Cost is about $1/hr.

### SSH Setup

Prime Intellect assigns a custom SSH port (not the default 22). This matters for both `ssh` and `scp`.

```bash
# Generate a dedicated keypair (do this once on your local machine)
ssh-keygen -t ed25519 -a 64 -f ~/.ssh/primeintellect_ed25519 -C "primeintellect"

# Add the public key in the Prime Intellect dashboard before deploying
cat ~/.ssh/primeintellect_ed25519.pub

# Connect (note the custom port)
ssh -i ~/.ssh/primeintellect_ed25519 -p <port> root@<ip>

# Copy files TO the node (note: scp uses uppercase -P for port)
scp -P <port> -i ~/.ssh/primeintellect_ed25519 local_file.py root@<ip>:/workspace/
```

### Install Everything

**Important: Install order matters.** Install all packages (including verifiers-rl) BEFORE starting vLLM. On our first attempt, we started vLLM first, then tried to compile flash-attn — this ate all available RAM and crashed the node so hard we couldn't even run `kill` or `ls`. We had to destroy the instance and start over.

```bash
# uv — fast Python package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create a virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# prime CLI — installs environments from the hub
uv tool install prime && uv tool update-shell

# Install the alphabet-sort environment (pinned version for reproducibility)
prime env install primeintellect/alphabet-sort@0.1.5

# Install verifiers (note: [all] extra doesn't exist on v0.1.10, but doesn't error)
uv pip install 'verifiers[all]'

# Install verifiers-rl from GitHub (not on PyPI)
# IMPORTANT: Install WITH dependencies — do NOT use --no-deps
# This will compile flash-attn from source (~15 min) but ensures all versions are compatible
# Make sure vLLM is NOT running during this step — flash-attn compilation needs the memory
git clone --depth 1 https://github.com/PrimeIntellect-ai/verifiers.git /tmp/vf-repo
uv pip install /tmp/vf-repo/packages/verifiers-rl

# wandb for monitoring (not bundled with verifiers — it's optional)
uv pip install wandb
```

**Why install WITH dependencies?** We originally tried `--no-deps` to skip flash-attn compilation. This was a mistake — it led to hours of debugging missing packages (deepspeed, liger-kernel), version mismatches (vllm 0.15 vs required 0.10), and broken NCCL communication. The 15-minute flash-attn compile is annoying but it ensures all versions are pinned correctly. Just make sure vLLM isn't running during install so the compile has enough memory.

**Why clone from GitHub?** `verifiers-rl` was never published to PyPI. The only way to install it is from the monorepo source at `packages/verifiers-rl/`. Using `--depth 1` keeps the clone fast (~12MB instead of full history).

## Part 2: Serving the Model

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve willcb/Qwen3-0.6B --enforce-eager --disable-log-requests
```

Wait for `Application startup complete.` — that means the server is ready at `http://0.0.0.0:8000/v1`.

### vllm serve vs vf-vllm: A Critical Distinction

For **evaluation**, `vllm serve` is fine. The model weights never change — you serve a model, run prompts, score outputs, done.

For **training**, you MUST use `vf-vllm` instead. Here's why: during GRPO training, the model weights update every step on GPU 1. But the vLLM server on GPU 0 is still serving the *old* weights. Without a way to sync, your rollouts would always come from the original untrained model — the training loop would never see improvement in its own generations.

`vf-vllm` is not just a convenience wrapper around `vllm serve`. It adds custom HTTP endpoints (`/get_world_size`, `/init_communicator`, `/update_weights`) that let the trainer push new weights directly into the running vLLM process over NCCL after each training step. The two GPUs are actively communicating:

```
GPU 0: vf-vllm (serves model + accepts weight updates from trainer)
  ↕ HTTP + NCCL weight sync
GPU 1: RLTrainer (trains model, pushes new weights after each step)
```

For eval:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve willcb/Qwen3-0.6B --enforce-eager --disable-log-requests
```

For training:
```bash
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests
```

The blog tutorial never explains this distinction — it just says "use vf-vllm" without saying why. If you use `vllm serve` for training, you'll get a 404 error on `/get_world_size` because regular vLLM doesn't have that endpoint.

This is also why `prime-rl` is simpler for training — it launches its own vLLM process with weight syncing built in. You never have to think about it.

### Why vLLM?

Instead of loading the model in Python (slow, blocks everything), vLLM gives you a high-throughput inference server. Both the evaluator and the trainer talk to it over HTTP. It handles batching, continuous generation, and serves an OpenAI-compatible API — so any tool that speaks OpenAI can use your local model.

### Why willcb/Qwen3-0.6B instead of Qwen/Qwen3-0.6B?

This is a critical gotcha. The original Qwen3 models strip `<think>` sections from messages when processing inputs between turns. In a multi-turn setting, this means the model loses its own reasoning from previous turns — it violates the "increasing context" requirement that GRPO training depends on.

The `willcb/Qwen3-0.6B` fork has a modified chat template that preserves think tokens. For eval-only it probably doesn't matter much, but for training it's essential.

### What about vf-vllm?

The `vf-vllm` command used to be a convenient wrapper around `vllm serve`. It has since moved to the optional `verifiers-rl` package and is no longer available with a basic `verifiers` install. Just use `vllm serve` directly — it does the same thing.

## Part 3: Configuration Files

### The Old Way: CLI Args + Python Endpoints

The original tutorial blog post uses command-line flags and a Python dictionary:

```python
# endpoints.py
ENDPOINTS = {
    "Qwen3-0.6B": {
        "model": "willcb/Qwen3-0.6B",
        "url": "http://0.0.0.0:8000/v1",
        "key": "EMPTY",
    },
}
```

```bash
vf-eval alphabet-sort -m Qwen3-0.6B -e "endpoints.py" -n 5 -r 3 -t 1024 \
  --save-dataset --save-to-hf-hub --hf-hub-dataset-name "user/dataset-name"
```

This works but it's fragile — long commands are easy to mistype, hard to share, and impossible to version-control meaningfully.

### The New Way: TOML Config Files

We separate concerns into two files:

```bash
mkdir -p configs
```

**configs/endpoints.toml** — Where are your models?

```toml
[[endpoint]]
endpoint_id = "Qwen3-0.6B"
model = "willcb/Qwen3-0.6B"
url = "http://0.0.0.0:8000/v1"
key = "EMPTY"
```

You can define multiple endpoints here — local vLLM, OpenAI API, any provider. The `endpoint_id` is a friendly name that other configs reference. `key = "EMPTY"` because vLLM doesn't need auth, but the OpenAI client still requires a key field.

**configs/eval.toml** — What do you want to evaluate?

```toml
[[eval]]
env_id = "alphabet-sort"
model = "Qwen3-0.6B"
num_examples = 5
rollouts_per_example = 3
max_tokens = 1024
save_results = true
save_to_hf_hub = true
hf_hub_dataset_name = "gustofied/Qwen3-0.6B-alphabet-sort-eval"
```

**Gotcha:** The field is `env_id`, not `env`. The error message tells you if you get it wrong.

You can add multiple `[[eval]]` blocks to run different evaluations in sequence. The double brackets `[[eval]]` mean "array of tables" in TOML — each block is one evaluation run.

## Part 4: Baseline Evaluation

```bash
export HF_TOKEN=your_token_here
vf-eval configs/eval.toml -e configs/endpoints.toml
```

### What's Happening Under the Hood

1. vf-eval reads both config files
2. Connects to the vLLM server at the endpoint URL
3. Loads the alphabet-sort environment (installed via `prime env install`)
4. Generates 5 prompts (random name lists to sort)
5. For each prompt, runs 3 rollouts (same prompt, different sampling)
6. Each rollout is a multi-turn conversation — the model sorts, gets new names, sorts again
7. The reward function scores each rollout using difflib similarity^4
8. Results are saved locally and pushed to HF Hub

### What's a Rollout?

One complete interaction between the model and the environment. With 3 rollouts per example, the same prompt runs 3 times with different random sampling. This gives you:
- **Variance estimates** — how consistent is the model?
- **The GRPO signal** — during training, the group mean vs individual rollout score tells the optimizer which responses were better than average

### Baseline Results

The base Qwen3-0.6B scores about **~0.4 mean reward**. It understands the task but makes frequent sorting mistakes. This is our "before" number.

## Part 5: Training with GRPO

### How GRPO Works

GRPO (Group Relative Policy Optimization) is how the model learns:

1. Take a prompt (a list of names to sort)
2. Generate N completions (default 8) using sampling — each one is slightly different
3. Score each completion with the deterministic reward function
4. Calculate the group mean score
5. Completions above the mean get positive advantage (model should do more of this)
6. Completions below the mean get negative advantage (model should do less of this)
7. Update weights accordingly

The key insight: you don't need a separate "critic" or "value network" — the group average IS your baseline. This makes GRPO simpler and more memory-efficient than PPO.

### The Verifiers Ecosystem Split

This is where things get confusing. The verifiers library has been reorganized:

| Package | Status | What it does |
|---------|--------|-------------|
| `verifiers` | Active, on PyPI | Environments, evaluation (`vf-eval`), core library |
| `verifiers-rl` | Legacy, NOT on PyPI | `vf.GRPOTrainer`, `vf.get_model_and_tokenizer`, `vf-vllm` |
| `prime-rl` | Active, recommended | Production async RL trainer, TOML-configured |

The tutorial blog post uses `vf.GRPOTrainer` which lives in `verifiers-rl`. But `verifiers-rl` isn't published to PyPI — you have to install it from GitHub:

```bash
uv pip install torch
uv pip install flash-attn --no-build-isolation
uv pip install "verifiers-rl @ git+https://github.com/PrimeIntellect-ai/verifiers.git#subdirectory=packages/verifiers-rl" --no-build-isolation
```

### The Training Script (Legacy Approach)

This follows the tutorial's approach using `vf.GRPOTrainer`:

```python
import verifiers as vf
import wandb


def main():
    wandb.login()

    model_name = "willcb/Qwen3-0.6B"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment("alphabet-sort")

    training_args = vf.grpo_defaults(run_name="alphasort-grpo-qwen-3")

    # Batch configuration
    training_args.per_device_train_batch_size = 8  # Prompts per GPU per step
    training_args.gradient_accumulation_steps = 8  # Steps before optimizer update
    # effective batch size = 8 * 8 = 64
    training_args.num_generations = 8  # Completions per prompt (GRPO group size)
    training_args.max_completion_length = 2048

    # Async generation — keeps GPU busy while environment processes
    training_args.num_batches_ahead = 1
    training_args.async_generation_timeout = 300.0
    training_args.max_concurrent = 1024

    training_args.max_steps = 1000

    # Monitoring
    training_args.logging_steps = 1
    training_args.log_completions = True
    training_args.report_to = "wandb"
    training_args.num_completions_to_print = 1

    # Saving — checkpoints every 100 steps, pushed to HF Hub
    training_args.output_dir = "./mymodel"
    training_args.overwrite_output_dir = True
    training_args.hub_model_id = "gustofied/Qwen3-0.6B-alphabet-sort-grpo"
    training_args.hub_strategy = "every_save"
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.save_total_limit = 1
    training_args.push_to_hub = True

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
```

Run on GPU 1 (GPU 0 is running vLLM):

```bash
export WANDB_API_KEY=your_key
export HF_TOKEN=your_token
CUDA_VISIBLE_DEVICES=1 python training_script.py
```

### The Modern Approach: prime-rl

The recommended way is `prime-rl`, which replaces the Python script with a TOML config:

```toml
model = "willcb/Qwen3-0.6B"
max_steps = 500
batch_size = 64
rollouts_per_example = 8

[sampling]
max_tokens = 2048

[[env]]
id = "primeintellect/alphabet-sort"

[wandb]
project = "alphabet-sort"
name = "qwen3-0.6b-alphabet-sort"
```

```bash
uv run prime-rl configs/train.toml
```

This handles the vLLM server, training loop, and orchestration automatically. No Python script needed. The TOML *is* the training config — same philosophy as using TOML for eval instead of CLI args.

### Training Parameters Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `per_device_train_batch_size` | 8 | Prompts loaded per GPU per step. Limited by VRAM |
| `gradient_accumulation_steps` | 8 | Accumulate gradients over 8 mini-batches before updating. Effective batch = 8 x 8 = 64 |
| `num_generations` | 8 | GRPO group size. 8 completions per prompt. The group mean becomes the baseline for advantage calculation |
| `max_completion_length` | 2048 | Max tokens per completion. Longer = more reasoning but more memory |
| `max_steps` | 1000 | Total optimizer steps. With batch 64, this sees 64,000 prompt-response pairs |
| `num_batches_ahead` | 1 | Async generation — prepare the next batch while training on current one |
| `save_steps` | 100 | Checkpoint every 100 steps. Pushed to HF Hub |

## Part 6: Evaluate the Fine-Tuned Model

After training completes:

1. Stop the old vLLM server (Ctrl+C)
2. Serve the fine-tuned checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve gustofied/Qwen3-0.6B-alphabet-sort-grpo --enforce-eager --disable-log-requests
```

3. Update endpoints:

```bash
cat > configs/endpoints.toml <<'EOF'
[[endpoint]]
endpoint_id = "Qwen3-0.6B-tuned"
model = "gustofied/Qwen3-0.6B-alphabet-sort-grpo"
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
```

4. Run evaluation:

```bash
vf-eval configs/eval-tuned.toml -e configs/endpoints.toml
```

### Expected Results

- **Before training:** ~0.4 mean reward
- **After training:** ~0.58+ mean reward (~43% improvement)

The model learns to sort more accurately and consistently across turns.

## Gotchas and Lessons Learned

| Problem | What happened | Fix |
|---------|--------------|-----|
| `vf-vllm` not found | Moved to `verifiers-rl` package | Use `vllm serve` directly |
| `TRANSFORMERS_CACHE` warning | Old env var name deprecated | Harmless, ignore it |
| scp silent failure | Wrong port (default 22 vs custom) | Use `-P <port>` (uppercase P) |
| scp permission denied | Missing SSH key | Add `-i ~/.ssh/primeintellect_ed25519` |
| `env` vs `env_id` in TOML | Wrong field name | The eval config field is `env_id` |
| Wrong TOML file | Saved eval config as endpoints.toml | Two separate files with different schemas |
| `No module named 'wandb'` | Not bundled with verifiers | `uv pip install wandb` separately |
| Qwen3 strips think tokens | Original model breaks multi-turn | Use `willcb/Qwen3-0.6B` fork |
| `verifiers-rl` not on PyPI | Legacy package, unpublished | Install from GitHub with subdirectory path |
| flash-attn build fails | Needs torch at build time | Install torch first, then `--no-build-isolation` |
| `vf.GRPOTrainer` is legacy | Docs recommend prime-rl | Works for learning, use prime-rl for production |
| `No module named 'deepspeed'` | verifiers-rl imports it unconditionally | `uv pip install deepspeed` (missed by `--no-deps`) |
| `No module named 'liger_kernel'` | verifiers-rl imports it unconditionally | `uv pip install liger-kernel` (missed by `--no-deps`) |
| 404 on `/get_world_size` | Used `vllm serve` instead of `vf-vllm` for training | Training requires `vf-vllm` for weight sync between GPUs |
| `GRPOTrainer` wrong arg order | Legacy wrapper passes `(model, processing_class, env, args)` but `RLTrainer` expects `(model, env, args)` | Use `RLTrainer` directly with keyword args |

## Resources

- [Environments Hub](https://app.primeintellect.ai/dashboard/environments)
- [Verifiers docs](https://verifiers.readthedocs.io/)
- [prime-rl GitHub](https://github.com/PrimeIntellect-ai/prime-rl)
- [anakin87's tutorial](https://huggingface.co/blog/anakin87/environments-hub) (the blog post this is based on)
- [Training docs](https://docs.primeintellect.ai/verifiers/training)
