# TODO: Extract and Modernize verifiers-rl

## The Idea

Take the valuable parts of verifiers-rl out of the verifiers monorepo subtree and make it a standalone, hackable package. Then upgrade it to work with modern TRL and vllm instead of being frozen at July 2025 versions.

## Current State

verifiers-rl lives nested inside the verifiers subtree:
- `terminal2F/external/verifiers/packages/verifiers-rl/`
- Also copied at `terminal2F/verifiers/packages/verifiers-rl/`

This is awkward for hacking on. It should be a top-level directory or its own repo.

## What to Extract

The valuable parts are small and well-defined:

1. **`orchestrator.py`** — the multi-turn conversation loop. Model responds, environment gives new input, model responds again, score the whole trajectory. This is the piece TRL doesn't have.

2. **`client.py`** — vLLM weight sync. Pushes updated weights from training GPU to inference GPU via NCCL after each step.

3. **`trainer.py`** — glue that connects TRL's GRPOTrainer to the orchestrator.

4. **`config.py`** — RLConfig (extends TRL's GRPOConfig with multi-turn fields).

## Modernization Path

- Port the orchestrator logic into TRL's `rollout_func` parameter (added recently, designed for custom rollout strategies)
- Use modern vllm directly (no vf-vllm wrapper — handle weight sync differently or use vllm's native mechanisms)
- Drop the `vllm>=0.10.0,<0.11.0` pin — run on latest vllm
- Drop flash-attn as a hard dependency (make it optional)
- Keep it as a thin layer: just multi-turn orchestration + verifiers environment integration

## Steps

1. Copy `verifiers-rl/` to top-level in terminal2F (out of the subtree)
2. Finish the tutorial first — understand what each piece does end to end
3. Read TRL's `rollout_func` docs and source
4. Rewrite the orchestrator as a rollout function
5. Test with alphabet-sort on a GPU node
6. Optionally: make it its own repo
