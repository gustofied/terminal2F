## How Advantages Update Weights

How the advantage signal flows through the loss function to actually change model weights. The mechanics of turning "good completion / bad completion" into gradient updates.

### The Goal

Each token the model generated has a logprob — how probable that token was when the model picked it. After scoring, you know the advantage (good or bad completion). The trainer's job:

- **Positive advantage** → push logprobs of those tokens **up** (make the model more likely to pick them again in similar contexts)
- **Negative advantage** → push logprobs of those tokens **down** (make the model less likely to pick them)

### How It Actually Happens

**Step 1: The trainer re-scores the same tokens.**

vLLM generated the completion with the old weights and recorded its logprobs (`sampling_logprobs`). Now the trainer feeds the same token sequence through its copy of the model and gets fresh logprobs (`trainer_logprobs`). This is not another rollout — it's just a forward pass on the same fixed token sequence to get logprobs under the current weights. These might already differ slightly because the weights have been updated from previous steps.

**Step 2: The loss function.**

For each completion token t (mask = 1):

```
ratio_t = exp(trainer_logprob_t - sampling_logprob_t)
loss_t  = -ratio_t × advantage
```

Total loss = sum of loss_t over all completion tokens, over all rollouts in the microbatch.

The ratio is a PPO-style importance-sampling correction (using the rollout policy as "old"). Some GRPO implementations skip the ratio if they treat updates as strictly on-policy, but many LLM RL stacks keep it with clipping for stability.

Let's unpack what this does with real numbers.

Say the model generated the token `"alice"` (ID 4417):

```
sampling_logprob = -0.3    (token probability ≈ 0.74)
trainer_logprob  = -0.35   (token probability ≈ 0.70)
advantage        = +0.4    (this was a good completion)

ratio = exp(-0.35 - (-0.3)) = exp(-0.05) = 0.95
loss  = -0.95 × 0.4 = -0.38
```

The derivative is `∂L/∂log π = -A × r = -0.4 × 0.95 = -0.38`. Since A > 0, gradient descent increases the logprob for `"alice"` — making the model more likely to pick it in this context. That's exactly what we want.

Now a bad completion with the token `"bob"` (ID 3382):

```
sampling_logprob = -1.2
trainer_logprob  = -1.1
advantage        = -0.4    (this was a bad completion)

ratio = exp(-1.1 - (-1.2)) = exp(0.1) = 1.10
loss  = -1.10 × (-0.4) = +0.44
```

The derivative is `∂L/∂log π = -A × r = -(-0.4) × 1.10 = +0.44`. Since A < 0, gradient descent decreases the logprob for `"bob"` — making the model less likely to pick it here.

The loss value itself (negative or positive) doesn't matter. What matters is the gradient direction, which the advantage controls:

```
∂L/∂log π = -A × r
```

- A > 0 → gradient descent increases log π (makes the sampled token more likely)
- A < 0 → gradient descent decreases log π (makes it less likely)

**Step 3: Clipping (the PPO part).**

The ratio can get extreme if the policy has shifted a lot. Clipping caps it:

```
clipped_ratio = clip(ratio, 1 - ε, 1 + ε)    # typically ε = 0.2
loss = -min(ratio × advantage, clipped_ratio × advantage)
```

This prevents any single update from moving the weights too aggressively. Clipping stops further movement in the direction that would increase the objective — for A > 0, it caps large increases in probability; for A < 0, it caps large decreases.

**Step 4: The loss mask.**

The loss is only computed on completion tokens (mask = 1). Prompt tokens are there for context but contribute zero to the loss. So when you sum up the total loss across the sequence, it's only over the tokens the model actually generated.

**Step 5: Backprop.**

The total loss (summed across all completion tokens, across all examples in the microbatch) gets backpropagated through the network. This produces gradients on every weight in the model. The optimizer (usually AdamW) uses those gradients to nudge the weights.

### What Changes in the Weights

This is the part that feels abstract. The model has billions of parameters — attention weights, MLP weights, embeddings. Backprop computes how much each weight contributed to the loss, and nudges it accordingly.

You don't directly say "make the logprob of token 4417 go up." Instead, the gradient flows backward through the entire network:

```
loss → output logits → final layer weights → ... → attention weights → ... → embeddings
```

Every weight gets a tiny nudge. The cumulative effect of all those nudges is that next time the model sees a similar prompt, the logit for `"alice"` will be slightly higher and the logit for `"bob"` will be slightly lower. That shifts the softmax, which shifts the logprobs, which shifts what the model samples.

### The Advantage is the Steering Wheel

The advantage doesn't change the loss function's shape — it **scales** it. The loss function always tries to push logprobs. The advantage controls **which direction and how hard**:

- Large positive advantage → strong push to increase these token logprobs
- Small positive advantage → gentle push
- Zero advantage → no push at all (this completion was average, nothing to learn)
- Negative advantage → push to decrease these token logprobs

That's why GRPO needs reward diversity in the group. If all K completions get the same reward, all advantages are zero, and the loss contributes nothing. No learning happens.

### One Completion, All Tokens Move Together

Every token in the completion gets the same advantage value. If the completion scored well, *all* its tokens get reinforced — even the "boring" ones like punctuation or formatting. The model can't tell which specific tokens were responsible for the good score. This is the **credit assignment** limitation of sequence-level reward.

Over many training steps and many examples, the signal averages out: tokens that consistently appear in good completions get reinforced, tokens that consistently appear in bad ones get suppressed. This is coarse credit assignment (sequence-level advantage applied to all tokens), but it works surprisingly well in practice because the model sees enough variation across the group to tease out what matters.
