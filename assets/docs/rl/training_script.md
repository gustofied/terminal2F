## Training Script: Legacy vs Corrected

The tutorial blog uses `vf.GRPOTrainer` with `grpo_defaults()`. This has issues. Here's the legacy version and what we actually use.

### Legacy (from the tutorial blog)

```python
import verifiers as vf
import wandb

def main():
    wandb.login()
    model_name = "willcb/Qwen3-0.6B"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment("alphabet-sort")

    training_args = vf.grpo_defaults(run_name="alphasort-grpo-qwen-3")
    training_args.per_device_train_batch_size = 8
    training_args.gradient_accumulation_steps = 8
    training_args.num_generations = 8
    training_args.max_completion_length = 2048
    training_args.max_steps = 1000
    training_args.logging_steps = 1
    training_args.report_to = "wandb"
    training_args.output_dir = "./mymodel"
    training_args.push_to_hub = True

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
    )
    trainer.train()
```

Problems with this:
- `vf.GRPOTrainer` is a wrapper with a broken positional arg order
- `grpo_defaults()` is deprecated, returns an `RLConfig` anyway
- Field names like `per_device_train_batch_size` and `num_generations` are old TRL names - `RLConfig` uses `micro_batch_size` and `rollouts_per_example`
- `use_lora` defaults to `True` (tutorial assumes full fine-tuning)
- `gradient_accumulation_steps` is hardcoded to 1 in `RLConfig.__post_init__`

### Corrected (what we actually run)

```python
from verifiers_rl.rl.trainer import (
    RLTrainer, RLConfig, get_model_and_tokenizer
)
import verifiers as vf
import wandb

def main():
    wandb.login()
    model_name = "willcb/Qwen3-0.6B"
    vf_env = vf.load_environment("alphabet-sort")

    args = RLConfig(
        run_name="alphasort-grpo-qwen-3",
        use_liger=True,
        use_lora=False,           # tutorial wants full fine-tuning
        batch_size=64,            # total rollouts per batch
        micro_batch_size=4,       # per GPU per step (reduced from 8, OOM)
        rollouts_per_example=8,   # GRPO group size
        max_seq_len=2048,
        max_steps=1000,
        max_concurrent=1024,
        generation_timeout=300.0,
        logging_steps=1,
        report_to="wandb",
        output_dir="./mymodel",
        overwrite_output_dir=True,
        hub_model_id="gustofied/Qwen3-0.6B-alphabet-sort-grpo",
        hub_strategy="every_save",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        push_to_hub=True,
    )

    model, tokenizer = get_model_and_tokenizer(
        model_name, use_liger=True
    )

    # Use RLTrainer directly, not the GRPOTrainer wrapper
    trainer = RLTrainer(
        model=model,
        env=vf_env,
        args=args,
        processing_class=tokenizer,
    )
    trainer.train()
```

Key differences:
- Import `RLTrainer` and `RLConfig` directly from `verifiers_rl`
- Use the correct field names (`micro_batch_size`, `rollouts_per_example`, `batch_size`)
- Explicitly set `use_lora=False` and `use_liger=True`
- Use keyword args to avoid positional arg confusion
- `micro_batch_size=4` instead of 8 (OOM fix on 48GB GPUs)

### Next step: TOML instead of Python

Even verifiers-rl supports TOML config via `vf-rl`. No Python script needed:

```toml
model = "willcb/Qwen3-0.6B"

[env]
id = "primeintellect/alphabet-sort"

[inference]
gpus = 1
args = { enforce_eager = true, disable_log_requests = true }

[trainer]
gpus = 1
```

```bash
vf-rl @ configs/train.toml
```

This spins up a tmux session with vf-vllm in the top pane and `vf-train` in the bottom, auto-assigns GPUs (inference gets GPU 0, trainer gets GPU 1). Same thing we did manually with two terminals.

For production, use `prime-rl` instead - same TOML idea but actively maintained, multi-node support, difficulty filtering, and more:

```bash
prime lab setup --prime-rl
uv run prime-rl configs/prime-rl/alphabet-sort.toml
```

### Results

- Base Qwen3-0.6B: ~0.06-0.16 reward
- After GRPO (step 300 checkpoint): 0.84 reward
- GPT baseline: 0.86 reward

Training collapsed after step ~250 (entropy exploded, mismatch_kl spiked). The step 300 checkpoint was from before the collapse. No KL penalty in verifiers-rl may have contributed.
