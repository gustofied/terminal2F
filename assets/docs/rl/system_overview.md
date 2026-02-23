## System Overview: Verifiers, Trainer, vLLM

How the three pieces connect, who talks to who, and why.

### The Three Pieces

```
┌─────────────────────────────────────────────────────┐
│  TRAINER PROCESS                                    │
│                                                     │
│  ┌─────────────┐       ┌──────────────────────┐     │
│  │   Trainer    │◄──────│    Orchestrator       │     │
│  │             │       │                      │     │
│  │ - loss      │       │ - calls env.generate()│     │
│  │ - backprop  │       │ - packages microbatches│    │
│  │ - weight    │       │ - computes advantages │     │
│  │   update    │       └──────────┬───────────┘     │
│  └──────┬──────┘                  │                 │
│         │                         │                 │
└─────────┼─────────────────────────┼─────────────────┘
          │                         │
          │ weight sync             │ env.generate()
          │ (NCCL)                  │
          ▼                         ▼
   ┌─────────────┐         ┌───────────────┐
   │    vLLM     │◄────────│   Verifiers   │
   │             │         │   (Env)       │
   │ - inference │ completions │            │
   │ - serves    │ token IDs   │ - multi-turn│
   │   model     │ logprobs    │   loop     │
   │             │─────────►│ - scoring    │
   └─────────────┘         └───────────────┘
```

### Who Talks to Who

**Orchestrator → Verifiers**: "Generate a batch of rollouts using this vLLM endpoint."

**Verifiers → vLLM**: "Give me completions for these prompts." (OpenAI-compatible API). This is the only thing that queries vLLM for generation. Not the trainer.

**Verifiers → Orchestrator**: Returns scored rollouts — token IDs, logprobs, rewards, trajectory structure.

**Trainer → vLLM**: "Here are updated weights." (NCCL GPU-to-GPU broadcast). This is the only thing that updates vLLM. Not verifiers.

**Orchestrator → Trainer**: Packages rollouts into microbatches (input_ids, loss_mask, sampling_logprobs, advantages).

### The Rollout (What Happens Inside env.generate)

For a single-turn environment:

```
Verifiers receives prompt from dataset
    │
    ▼
Sends prompt to vLLM ──────────► vLLM generates completion
                                       │
    ◄──────────────────────────────────┘
    │  returns: text + token IDs + logprobs
    ▼
Reward functions score the completion
    │
    ▼
Returns: scored rollout (tokens, logprobs, reward)
```

For multi-turn (like email-to-cc-bcc):

```
Verifiers receives prompt + follow_ups + ground_truths from dataset
    │
    ▼
Turn 1: sends prompt to vLLM ──► vLLM generates answer
    ◄───────────────────────────┘
    │
    ▼
env_response() returns question_2
    │
    ▼
Turn 2: sends prompt + answer_1 + question_2 to vLLM ──► vLLM generates answer
    ◄──────────────────────────────────────────────────┘
    │
    ▼
env_response() returns question_3
    │
    ▼
Turn 3: sends full conversation to vLLM ──► vLLM generates answer
    ◄─────────────────────────────────────┘
    │
    ▼
max_turns reached → stop
    │
    ▼
Reward functions score all turns against ground_truths
    │
    ▼
Returns: scored rollout with full trajectory
```

The env controls the conversation flow. vLLM just generates when asked. The trainer never sees any of this — it only gets the packaged result.

### The Training Step

```
Step N:
    │
    ├── 1. Orchestrator waits for rollouts from verifiers
    │
    ├── 2. Orchestrator computes advantages (GRPO: reward - group_mean)
    │
    ├── 3. Orchestrator packages microbatches:
    │       - input_ids (prompt + completion tokens)
    │       - loss_mask (0 for prompt/env, 1 for model output)
    │       - sampling_logprobs (from vLLM at generation time)
    │       - advantages (per rollout, repeated across tokens)
    │
    ├── 4. Trainer forward pass on input_ids → trainer_logprobs
    │
    ├── 5. Loss = -ratio × advantage (masked to completion tokens)
    │
    ├── 6. Backprop → update trainer weights
    │
    ├── 7. Wait for vLLM to finish any in-flight generation
    │
    ├── 8. Sync weights to vLLM (NCCL broadcast)
    │
    └── 9. Orchestrator submits next batch to verifiers → Step N+1
```

### Weight Sync Timing

The trainer coordinates everything because it owns the weights. The sequence matters:

```
 generate with weights v1    train on rollouts    generate with weights v2
├──────────────────────────┤├───────────────────┤├──────────────────────────┤
                            ▲                   ▲
                            │                   │
                       wait for gen          sync v2
                       to finish             to vLLM
```

You can't sync while vLLM is generating — that would corrupt in-flight inference. You can't generate while syncing — you'd get mixed weights. So the trainer enforces: finish generating → sync → start generating.

### What the Config Connects

The trainer config ties everything together:

```toml
model = "Qwen/Qwen3-4B"           # what vLLM loads
batch_size = 256                    # how many rollouts per step
rollouts_per_example = 8            # K in GRPO (group size)

[sampling]
max_tokens = 512                    # vLLM generation limit

[[env]]
id = "email-to-cc-bcc"             # which verifiers env to use
args = { max_turns = 3 }           # passed to load_environment()
```

The trainer reads this, spins up vLLM with the model, creates the orchestrator, loads the environment, and starts the loop. Everything flows from the config.

### "Trainer-Agnostic" (What That Actually Means)

Verifiers has no trainer dependency in its code. It just needs an OpenAI-compatible endpoint for generation. But any trainer that wants to use verifiers needs to build an orchestrator layer that:

1. Calls `env.generate()` with the vLLM client
2. Parses the trajectory structure (token IDs, logprobs, turn boundaries)
3. Builds loss masks (model tokens = 1, prompt/env response = 0)
4. Handles weight sync timing around generation

Every trainer that supports verifiers (prime-rl, t2f-trainer, Tinker, SkyRL, rLLM) had to build this adapter. The env and reward functions are portable. The orchestrator glue is not.
