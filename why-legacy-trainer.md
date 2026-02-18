# Why We're Using the Legacy Trainer

## The Decision

We're using `vf.GRPOTrainer` from the unpublished `verifiers-rl` package — the same approach from anakin87's blog tutorial — instead of the recommended `prime-rl`. This is a deliberate choice.

## Why Not prime-rl?

prime-rl is the "right" answer according to the docs. It's maintained, scales to multi-node, and uses clean TOML configs. But:

1. **It's a black box.** You run `uv run prime-rl configs/train.toml` and it launches a tmux session with a trainer, orchestrator, and inference server. Great for production, bad for learning. You can't step through what's happening. You can't see where the GRPO advantage is computed, how rollouts are collected, or how the model weights get updated.

2. **It's tightly coupled to Prime's ecosystem.** `prime lab setup` scaffolds their entire workflow. The TOML config format is prime-rl specific — no other trainer reads it. If you learn prime-rl, you've learned prime-rl. If you learn the Python training script, you've learned how GRPO training works in general.

3. **It's overkill for 0.6B on 2 GPUs.** prime-rl was built for INTELLECT-2 (32B parameters, distributed across many nodes). We're training a tiny model on a single node. The complexity isn't justified.

## Why the Legacy Trainer?

1. **It's Python you can read.** The training script is ~50 lines. You can see every parameter, understand what each one does, and modify it. There's no abstraction hiding the training loop.

2. **It matches every tutorial.** The blog post, the README examples, community posts — they all use `vf.GRPOTrainer`. When you're learning, being able to cross-reference with existing material matters.

3. **It teaches transferable concepts.** The parameters (`per_device_train_batch_size`, `gradient_accumulation_steps`, `num_generations`) are the same across all GRPO implementations — TRL, SkyRL, OpenRLHF. Learn them once in a readable Python script, apply them anywhere.

4. **It handles multi-turn.** This is the key technical reason. Alphabet-sort is a multi-turn environment. TRL's GRPOTrainer (the maintained HuggingFace version) is single-turn — it generates one completion and scores it. The verifiers-rl trainer manages the conversation loop: model responds, environment gives new input, model responds again, score the whole trajectory. Without it, we'd have to reimplement that loop ourselves.

## The Pain We Accepted

Using verifiers-rl comes with real costs:

- **Not on PyPI** — had to `git clone` the whole repo and install from a subdirectory
- **Depends on flash-attn** — which compiles CUDA kernels from source and nearly crashed our first node. We bypass this with `--no-deps`
- **Not maintained** — if it breaks with a future verifiers update, nobody's fixing it
- **Confusing error messages** — verifiers tells you "install as verifiers-rl" but doesn't tell you it's not on PyPI

## What We Actually Installed

```
verifiers (v0.1.10, from PyPI)
    └── environments, eval, vf-eval CLI

verifiers-rl (v0.1.0, from GitHub clone)
    └── vf.GRPOTrainer, vf.get_model_and_tokenizer, vf.grpo_defaults
    └── internally wraps TRL's GRPOTrainer

trl (v0.28.0, from PyPI)
    └── the actual GRPO implementation that verifiers-rl builds on

accelerate, peft, deepspeed (from PyPI)
    └── distributed training, LoRA support, and ZeRO optimization
    └── deepspeed is required — verifiers-rl imports it at startup even for single-GPU runs

vllm (from PyPI)
    └── model serving (installed separately since verifiers[all] extra doesn't exist)

wandb (from PyPI)
    └── training curve monitoring
```

The irony: `verifiers-rl` is a wrapper around TRL's GRPOTrainer that adds multi-turn environment support. The core GRPO math comes from TRL. The value verifiers-rl adds is the conversation loop orchestration — and that's exactly the part that's now abandoned.

## The Version Pinning Problem

`verifiers-rl` pins `vllm>=0.10.0,<0.11.0` (July–October 2025). Meanwhile vLLM shipped six major versions in seven months:

| Version | Date |
|---------|------|
| v0.10.0 | Jul 2025 |
| v0.11.0 | Oct 2025 |
| v0.12.0 | Dec 2025 |
| v0.13.0 | Dec 2025 |
| v0.14.0 | Jan 2026 |
| v0.15.0 | Jan 2026 |
| v0.16.0 | Feb 2026 |

vLLM moves fast and breaks APIs between majors — imports get renamed, endpoints change, argument parsers get refactored. `verifiers-rl` was abandoned in November 2025 (the v0.1.7 split), so its vLLM pin is frozen 4+ months behind. Installing vLLM normally gives you 0.15+ which immediately breaks `vf-vllm` (`FlexibleArgumentParser` was removed from `vllm.utils`).

The fix is downgrading: `uv pip install "vllm>=0.10.0,<0.11.0"`. But this means you're running a 6-month-old inference engine with known bugs and missing optimizations, just because the trainer wrapper was never updated. This is the real cost of using unmaintained software — you're not just frozen on the trainer, you're frozen on everything it touches.

## What We Did Wrong

The biggest mistake was installing verifiers-rl with `--no-deps` to skip flash-attn compilation. This avoided one problem but created ten more — every missing dependency (deepspeed, liger-kernel) was discovered one at a time through runtime crashes, and when we manually installed them we got incompatible versions.

The `--no-deps` flag skipped vllm version pinning too. verifiers-rl needs `vllm>=0.10.0,<0.11.0`, but we had 0.15.1 installed. After downgrading vllm, the flash-attn binary was compiled against the wrong torch, breaking it again. Then we disabled flash-attn with `use_liger=False` and `attn_implementation="eager"`, which worked but may have contributed to the NCCL hang — the version soup of vllm 0.10, newer torch, newer NCCL was untested territory.

**What we should have done:**

```bash
# On a fresh node, BEFORE starting vLLM (so flash-attn compile has memory):
git clone --depth 1 https://github.com/PrimeIntellect-ai/verifiers.git /tmp/vf-repo
uv pip install /tmp/vf-repo/packages/verifiers-rl
# Let it install everything: vllm 0.10.x, flash-attn, deepspeed, liger-kernel
# Wait 15 min for flash-attn to compile
# THEN start vf-vllm and training
```

One install command. All versions pinned correctly by the package. The flash-attn compile takes time but at least everything is compatible afterward. The tutorial's setup is actually simple — we made it complicated by trying to shortcut around flash-attn.

**Lesson:** When using abandoned packages with specific version requirements, install WITH dependencies. Fighting version mismatches one by one is always worse than waiting for a long compile.

## When to Switch

Once you understand how GRPO training works from running this script, the natural next step is:

- **prime-rl** if you're staying in the Prime Intellect ecosystem and want scale
- **TRL directly** if you're working on single-turn tasks (math, coding, formatting)
- **Write your own rollout loop** if you need multi-turn with a maintained trainer — take the conversation management from verifiers-rl's source and plug it into TRL's `rollout_func` parameter
