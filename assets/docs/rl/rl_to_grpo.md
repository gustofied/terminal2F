## From RL to GRPO: How LLMs Learn from Rewards

The path from basic RL to what we actually run. Covers policy gradients, logits, logprobs, baselines, PPO, GRPO, and the normalization fixes (Dr.GRPO/DAPO).

### RL Objects, Translated to LLMs

Standard RL has states, actions, policies, trajectories, and rewards. In an LLM:

- **State** — the prompt + tokens generated so far
- **Action** — the next token
- **Policy** — softmax over logits (probability distribution over the full vocabulary for the next token)
- **Trajectory** — one full completion (a sequence of tokens from start to stop)
- **Reward** — a score for the whole completion (from a verifier, unit tests, exact match, reward model)

The goal: adjust weights so the model generates higher-reward completions more often.

### Logits, Softmax, Logprobs

The model outputs a raw score for every token in the vocabulary. These are **logits** — unnormalized, can be any real number.

```
logits for next token: [2.1, 0.5, -1.3, 0.8, ...]  (one per vocab entry, ~100k values)
```

**Softmax** turns logits into a probability distribution (positive, sums to 1):

```
probs: [0.74, 0.15, 0.02, 0.08, ...]
```

**Logprobs** are the log of those probabilities:

```
logprobs: [-0.30, -1.90, -3.91, -2.53, ...]
```

Why log? A completion's total probability is the product of all its token probabilities. Products of small numbers get tiny fast. Logs turn products into sums, which are numerically stable and easier to differentiate:

```
p(completion) = p(token_1) × p(token_2) × ... × p(token_n)
log p(completion) = log p(token_1) + log p(token_2) + ... + log p(token_n)
```

This sum of logprobs is what connects "how likely is this completion" to gradient updates.

### The Policy Gradient (Core Idea)

The fundamental equation that makes all of this work:

```
gradient = E[R(completion) × sum of ∇ log π(token_t | context)]
```

In plain terms:

- Generate a completion, get a reward
- For each token in that completion, compute the gradient of its logprob
- Scale by the reward
- Update weights in that direction

**High reward** → increase logprobs of those tokens → model generates similar completions more often.

**Low reward** → decrease logprobs → model avoids similar completions.

This is REINFORCE — the simplest policy gradient. It works but it's noisy.

### The Variance Problem

REINFORCE uses the raw reward to scale gradients. Problem: if all your rewards are between 0.7 and 0.9, even the "bad" completions get positive gradient. The model reinforces everything, just at different magnitudes. Learning is slow and noisy.

**Solution: baselines and advantages.**

Instead of "this completion got reward 0.8," ask "was this completion better or worse than expected?"

```
advantage = reward - baseline
```

Now:
- Advantage > 0 → better than expected → reinforce
- Advantage < 0 → worse than expected → suppress

The baseline can be computed different ways. That's where PPO and GRPO diverge.

### PPO (Actor-Critic)

PPO uses a **critic** — a separate neural network that estimates "how much reward should I expect from this state?" That estimate is the baseline.

```
advantage = reward - critic_estimate
```

PPO also introduces a **ratio** between new and old policy:

```
ratio = π_new(token | context) / π_old(token | context)
```

And **clipping** — if the ratio gets too far from 1.0, cap it. This prevents any single update from changing the policy too much (trust region).

```
loss = min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)
```

**What PPO buys you:** stable training, even with multiple optimization steps per batch.

**What it costs:** a second model (the critic) that needs its own training, memory, and tuning. For large LLMs, that's significant overhead.

### RLHF vs RLVR

Two sources of reward:

**RLHF** (RL from Human Feedback) — a learned reward model trained on human preferences ("which response is better?"). Used for chat alignment. Prone to reward hacking because the reward model is imperfect.

**RLVR** (RL from Verifiable Rewards) — a deterministic verifier. Did the code pass unit tests? Is the math answer correct? Does the JSON parse? Cleaner signal, harder to hack.

Math and code are popular RLVR domains because "right or wrong" is objective.

### GRPO (Critic-Free)

GRPO removes the critic entirely. Instead of a learned baseline, it uses the **group average** as the baseline.

For each prompt:

1. Sample **K completions** (typically 4–16)
2. Score each with the verifier → rewards r_1, r_2, ..., r_K
3. Compute advantage per completion: `A_i = r_i - mean(r_1..r_K)`
4. Update: increase logprobs for above-average completions, decrease for below-average

```
prompt: "Assign recipients to To, CC, BCC"

completion_1: {"to": ["alice"], "cc": ["bob"], "bcc": []}     → reward 0.9
completion_2: {"to": ["alice", "bob"], "cc": [], "bcc": []}   → reward 0.4
completion_3: {"to": ["alice"], "cc": [], "bcc": ["bob"]}     → reward 0.6
completion_4: {"to": [], "cc": ["alice", "bob"], "bcc": []}   → reward 0.1

group mean = 0.5

advantages: [+0.4, -0.1, +0.1, -0.4]
```

Completion 1 gets reinforced the most. Completion 4 gets suppressed the most.

**Key requirement:** reward diversity within the group. If all K completions score the same, all advantages are zero and nothing is learned. This is why the To/CC learnability issue matters — if the model can't consistently distinguish the right answer, rewards become random across the group and advantages wash out.

**Tradeoff vs PPO:**

- No critic → simpler, less memory, no critic tuning
- But you need more rollouts (larger K) to get a stable baseline
- Works well when rewards are clean (RLVR) and sampling is cheap (vLLM)

### KL Regularization

Both PPO and GRPO often add a penalty for drifting too far from a reference policy (usually the SFT model you started from):

```
total_loss = policy_loss + β × KL(π_current || π_reference)
```

Without this, the model can collapse — finding a single reward-hacking pattern and repeating it. The KL penalty says "stay close to the original model's distribution." It's the practical trust region for LLM RL.

### Dr.GRPO and DAPO (Normalization Fixes)

The core GRPO idea is sound, but practical training can go sideways due to how you normalize and aggregate losses. These variants fix specific pathologies:

**Std normalization in advantage.** Some GRPO implementations divide advantage by the group's standard deviation: `A_i = (r_i - mean) / std`. Problem: groups where rewards happen to have small variance get amplified. A group of all-0.8 completions with one 0.9 outlier gets massive advantage despite being boring. Dr.GRPO says: drop the std normalization, just use `r_i - mean`.

**Length normalization in loss.** If you divide the loss by the completion's token count, shorter completions get stronger per-token gradient. The model learns "be brief" regardless of correctness. Dr.GRPO uses a fixed constant (e.g., max sequence length) instead of actual length. Removes the incentive to game length.

**DAPO** adds further recipes: token-level loss aggregation, overlong reward shaping (give partial credit instead of zero for truncated completions), and filtering out groups with zero variance. All trying to stabilize long reasoning traces.

**The meta-point:** most training instability in GRPO comes from normalization and aggregation choices, not from the policy gradient itself. The math is fine — it's the engineering details that trip people up.

### SFT vs PPO vs GRPO (Mental Model)

- **SFT** — "imitate these target tokens." Supervised, no reward signal. Fast, simple, but limited to the quality of your training data.
- **PPO** — "optimize reward with a critic + clipped updates." Powerful but heavy. Need a critic model, careful tuning, more memory.
- **GRPO** — "optimize reward by comparing multiple completions per prompt, no critic." Lighter, works great with clean verifier rewards. The sweet spot for RLVR tasks.
- **Dr.GRPO / DAPO** — "GRPO, but fix the normalization bugs that cause length bias and difficulty bias."

### What "Learning" Actually Means Here

RL doesn't label the right next token like SFT does. It **reweights which trajectories the model tends to sample:**

1. Generate completions with current weights
2. Score them
3. Compute advantages (better or worse than the group)
4. Shift token probabilities so good completions become more likely
5. Repeat

Over time, the model's distribution shifts — it samples high-reward completions more often, not because it was shown the "right answer," but because it tried many things and got feedback on what worked.
