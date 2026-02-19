## How GRPO Works — Example

The model tries it 3 times independently:

| | Answer | Correct? | Reward |
|---|---|---|---|
| Rollout 1 | a, b, c, d | yes | 1.0 |
| Rollout 2 | a, b, d, c | no | 0.0 |
| Rollout 3 | a, b, c, d | yes | 1.0 |

Average reward for this question: (1.0 + 0.0 + 1.0) / 3 = **0.67**

Now the advantage — how each rollout compares to the group mean:

- **Rollout 1**: 1.0 - 0.67 = **+0.33** (better than average)
- **Rollout 2**: 0.0 - 0.67 = **-0.67** (worse than average)
- **Rollout 3**: 1.0 - 0.67 = **+0.33** (better than average)

**Why this matters for RL:** the model can look at rollout 1 vs rollout 2 and ask "what did I do differently?" The positive advantage says "do more of this", negative says "do less of this." It learns from its own successes and failures on the same problem.

This is the core idea behind **GRPO** (Group Relative Policy Optimization) — you don't need a separate reward model, you just compare the rollouts against each other within the group.
