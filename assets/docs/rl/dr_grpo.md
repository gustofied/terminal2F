## Dr.GRPO: What It Fixes

Dr.GRPO (from DeepSeek-R1) makes two adjustments to standard GRPO. Both are about fairness — making sure no single group or response dominates the gradient just because of scale.

### 1. Group Std Normalization

Standard GRPO: `advantage = reward - group_mean`

Dr.GRPO: `advantage = (reward - group_mean) / group_std`

**The problem:** Groups with high reward variance produce larger advantages and dominate the gradient.

Example — two groups, 4 rollouts each:

- Group A: rewards `[0, 0, 0, 1]` — mean 0.25, advantages `[-0.25, -0.25, -0.25, 0.75]`
- Group B: rewards `[0.4, 0.5, 0.5, 0.6]` — mean 0.5, advantages `[-0.1, 0, 0, 0.1]`

Group A's advantages are 7x larger. Without normalization, the model mostly learns from Group A (the easy question where one rollout got lucky) and ignores Group B (where the model is actually making steady progress with nuanced signal).

Dividing by std scales both groups to similar magnitude — every prompt contributes equally to learning.

**Edge case:** If all rollouts in a group score the same, std is 0. These groups get skipped (advantage = 0), which is correct — zero variance means no learning signal.

**Where this lives in t2f-trainer:** `orchestrator.py`, where we compute advantages from raw rewards.

### 2. Response Length Normalization

Standard GRPO sums the loss over all tokens. Every token gets the same advantage, so longer responses have more impact on the gradient.

**The problem:** Within a group of rollouts for one prompt:

- Rollout 1: 400 tokens, reward 0
- Rollout 2: 50 tokens, reward 1
- Rollout 3: 200 tokens, reward 0

Without length normalization: 600 tokens of negative gradient vs 50 tokens of positive. The long wrong answer drowns out the short correct one. The model learns "don't be verbose" instead of "don't be wrong."

With length normalization: each rollout's loss is divided by its token count, so each response contributes equally regardless of length. Two wrong signals vs one right signal, weighted fairly.

**Where this lives in t2f-trainer:** `trainer.py` line 271, where the loss is computed:

```python
loss = (-importance_ratio * advantages)[keep_mask].sum()
```

This `.sum()` over all tokens is where length bias enters. Length normalization would divide each response's contribution by its token count before summing.
