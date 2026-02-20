## Training Architecture: Who Does What

Four pieces, each with one job:

### vLLM (Inference Server)
- Hosts the model as an OpenAI-compatible API
- Serves completions when asked
- Accepts weight updates from the trainer via NCCL

### Verifiers (Environment + Scoring)
- Calls vLLM to generate completions
- Manages multi-turn conversation loops (prompt → response → environment feedback → response → ...)
- Scores rollouts with rubrics
- Returns rollouts with rewards

### Trainer (Weight Updates)
- Receives rollouts and rewards from verifiers
- Computes advantages from the rewards (using whatever method the trainer chooses — GRPO, REINFORCE, etc.)
- Computes loss, updates model weights
- Pushes new weights back to vLLM
- Never serves completions, never scores anything

### Orchestrator (Glue)
- Lives inside the trainer (`orchestrator.py`)
- Calls `env.generate()` on verifiers with the vLLM client
- Processes the results into microbatches (token IDs, masks, logprobs, advantages)
- Feeds microbatches to the trainer's training loop

### The Loop

```
vLLM serves model
    ↓
Verifiers calls vLLM, gets completions, scores them
    ↓
Orchestrator packages scored rollouts into microbatches
    ↓
Trainer computes advantages, does weight update
    ↓
Weight sync pushes new weights to vLLM
    ↓
repeat
```

### Where Advantages Get Computed

This depends on which trainer you use:

**t2f-trainer / verifiers-rl** — verifiers computes advantages itself in `rubric.py` (`reward - mean(reward)`, no std normalization) and passes them pre-computed to the trainer. The trainer just uses them directly. This is opinionated — you're locked into verifiers' advantage formula.

**prime-rl, SkyRL, rLLM, etc.** — verifiers only returns rollouts + reward scores. The trainer computes its own advantages from the raw rewards using whatever algorithm it wants. More flexible — the trainer owns the full RL math.

### Why Verifiers is Trainer-Agnostic

Verifiers just needs an OpenAI-compatible endpoint. It doesn't care who's hosting it or who's consuming the results. Any trainer can plug in — it just needs to:

1. Spin up vLLM (or similar) with the model
2. Point verifiers at it
3. Consume the rollouts and rewards
4. Compute advantages, update weights, sync back to vLLM
