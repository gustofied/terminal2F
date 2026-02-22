## GRPO Training Loop: What Gets Fed to the Trainer

How data flows from vLLM generation through scoring to weight updates. Covers tokens, logprobs, masks, advantages, and the loss calculation.

### LLM Tokens vs Compiler Tokens

In compilers, a token is a typed symbolic unit — `IDENT("foo")`, `PLUS`, `SEMICOLON`. It carries structure and meaning. The lexer matches regex patterns against a grammar.

In LLMs, a token is just an integer index into a vocabulary table. No types, no grammar. The tokenizer chops text into subword pieces based on frequency:

```
"Hello world" → ["Hello", " world"] → [15496, 995]
```

Built via BPE (Byte Pair Encoding): start with individual bytes, repeatedly merge the most frequent adjacent pair until you hit ~100k vocabulary entries. Common words get one token, rare words get split:

```
"John"     → [2198]                    (common, one token)
"Zygmunt"  → [31849, 4452, 1628]       (rare, three tokens)
```

Two names can share subword tokens — "Hans" and "Hannah" might both start with a `" Han"` token but diverge after that. It's frequency-driven compression, closer to Huffman coding than lexical analysis.

### One vLLM Call, Two Outputs

vLLM generates a completion and returns both forms in a single response:

```python
# Token IDs — integers, for the trainer
prompt_ids     = response.prompt_token_ids          # [1042, 553, 8821]
completion_ids = response.choices[0].token_ids       # [4417, 229, 1105]

# Log probs — per completion token, for the trainer
completion_logprobs = [token.logprob for token in logprobs_content]  # [-0.3, -1.2, -0.1]

# Text — detokenized string, for verifiers/reward functions
text = response.choices[0].message.content           # '{"to": ["alice@acme.com"]}'
```

No re-tokenization needed. The token IDs and text are the same content in two representations. Token IDs go to the trainer, text goes to the reward functions (which need to parse JSON, compare strings, etc.).

Code: `verifiers/clients/openai_chat_completions_client.py`, lines 439–478.

### What the Trainer Receives

Four things per training example, packaged into a `Microbatch` by the orchestrator:

**1. `input_ids`** — the full token sequence, prompt + completion concatenated:

```
[1042, 553, 8821, 4417, 229, 1105]
 |--- prompt ---|  |-- completion --|
```

The trainer needs the full sequence because each token's probability depends on everything before it. You can't evaluate the completion without the prompt as context.

**2. `loss_mask`** — same length, binary. 0 = prompt (context only), 1 = completion (train on this):

```
[0, 0, 0, 1, 1, 1]
```

Built automatically by verifiers the moment vLLM returns — it knows the boundary because vLLM returns `prompt_token_ids` and `token_ids` separately. In multi-turn, environment responses between turns also get masked to 0:

```
[user prompt] [model turn 1] [env response] [model turn 2]
     0              1              0               1
```

Only the model's own outputs get trained on.

**3. `sampling_logprobs`** — the log probability vLLM assigned to each token when it generated them. These are the "old policy" probabilities:

```
[0.0, 0.0, 0.0, -0.3, -1.2, -0.1]
 |-- zeros for prompt --|  |-- completion logprobs --|
```

Log probs are the log of the softmax of the raw logits:

```
logits:   [2.1, 0.5, -1.3, ...]   ← raw scores for every vocab token
              ↓ softmax
probs:    [0.74, 0.15, 0.02, ...]  ← probabilities (sum to 1)
              ↓ log
logprobs: [-0.3, -1.9, -3.9, ...]  ← stored for the sampled token
```

Open-weight models on vLLM give full access to logits, so logprobs are computed directly. No API restrictions.

**4. `advantages`** — a single scalar (reward minus group mean), repeated for every token:

```
[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
```

Computed in the orchestrator from raw rewards:

```python
for g in range(0, len(raw_rewards), rollouts_per_example):
    group = raw_rewards[g : g + rollouts_per_example]
    group_mean = sum(group) / len(group)
    advantages.extend(r - group_mean for r in group)
```

GRPO generates N completions per prompt. If this completion scored 0.8 and the group mean was 0.5, the advantage is +0.3 (better than average, reinforce it). If it scored 0.2, advantage is -0.3 (worse than average, suppress it).

Code: `t2f_trainer/rl/trainer/orchestrator.py`, lines 253–258.

### The Loss Calculation

The trainer runs one forward pass on `input_ids`, gets its own log probs (`trainer_logprobs`), then:

```
ratio = exp(trainer_logprob - sampling_logprob)
loss = -ratio × advantage
```

- **Positive advantage** → negative loss → gradient makes these tokens more probable
- **Negative advantage** → positive loss → gradient makes these tokens less probable
- **Ratio** measures how much the policy has shifted — PPO clipping caps it so updates don't overshoot

The `loss_mask` ensures backprop only flows through completion tokens.

Code: `t2f_trainer/rl/trainer/trainer.py`, lines 260–270.

### The Full Loop

```
1. vLLM generates N completions per prompt (token IDs + logprobs + text)
2. Verifiers scores them with reward functions (text → reward scalar)
3. Orchestrator computes advantages (reward - group_mean)
4. Orchestrator packages microbatches (input_ids, loss_mask, sampling_logprobs, advantages)
5. Trainer forward pass → trainer_logprobs on same token sequences
6. Loss = -ratio × advantage, masked to completion tokens only
7. Backprop → update trainer weights
8. NCCL broadcast → sync weights to vLLM (GPU-to-GPU, every step)
9. Back to 1
```

### Two Copies, One Model

There are two copies of the same weights in GPU memory:

- **Trainer copy** (transformers/deepspeed) — where gradients flow and weights get updated
- **Inference copy** (vLLM) — generates completions fast, receives weight syncs

Same architecture, same weights after sync. Two copies because generation and gradient computation have different computational patterns and can't efficiently share a single loaded model. The sync happens every training step via NCCL broadcast — not at checkpoints. Checkpoints are periodic saves to disk for crash recovery and HuggingFace upload.

Code: `t2f_trainer/rl/trainer/trainer.py`, `update_vllm()` at line 319.
