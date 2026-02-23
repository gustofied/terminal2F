## Chat Templates: API → Template → Token IDs

Three representations of the same conversation, each one step closer to what the model actually sees.

### Step 1: Message List (What You Write)

The API format. Structured, readable, model-agnostic.

```json
[
  {"role": "system", "content": "You assign email recipients."},
  {"role": "user", "content": "Subject: Q3 deadline extension\n\nHi team..."},
  {"role": "assistant", "content": "{\"to\": [\"sarah@acme.com\"], \"cc\": [...]}"},
  {"role": "user", "content": "Subject: Re: Q3 deadline — escalated\n\nFollowing up..."},
  {"role": "assistant", "content": "{\"to\": [...], \"cc\": [...], \"bcc\": [...]}"}
]
```

This is what verifiers sends to vLLM. This is what you write in your env. You never think about templates here.

### Step 2: Template-Rendered String (What the Tokenizer Builds)

The tokenizer applies a Jinja template (stored in `tokenizer_config.json` on HuggingFace) and produces a raw string. Every model family has a different template.

**Qwen (ChatML):**
```
<|im_start|>system
You assign email recipients.<|im_end|>
<|im_start|>user
Subject: Q3 deadline extension

Hi team...<|im_end|>
<|im_start|>assistant
{"to": ["sarah@acme.com"], "cc": [...]}<|im_end|>
<|im_start|>user
Subject: Re: Q3 deadline — escalated

Following up...<|im_end|>
<|im_start|>assistant
{"to": [...], "cc": [...], "bcc": [...]}<|im_end|>
```

**Llama 3:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You assign email recipients.<|eot_id|><|start_header_id|>user<|end_header_id|>

Subject: Q3 deadline extension

Hi team...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"to": ["sarah@acme.com"], "cc": [...]}<|eot_id|>
```

**Mistral:**
```
[INST] Subject: Q3 deadline extension

Hi team... [/INST]{"to": ["sarah@acme.com"], "cc": [...]}[INST] Subject: Re: Q3 deadline...  [/INST]
```

Same message list, completely different strings. The model was trained on its specific format — feed it the wrong one and it still runs, it just performs worse because the token patterns are unfamiliar.

### Step 3: Token IDs (What the Model Sees)

The rendered string gets tokenized into integer IDs. This is the only thing that enters the model.

```
[151644, 8948, 198, 2610, 5765, ..., 151645, 198, 151644, 872, 198, ...]
```

Important: those delimiter strings like `<|im_start|>` are **single tokens** — `<|im_start|>` is token `151644` in Qwen, not a sequence of characters. The tokenizer has them in its vocabulary as special tokens.

The model outputs logits over this same vocabulary at each position. Softmax gives probabilities, log gives log-probs. Token IDs in, logits over token IDs out.

### Why This Matters for Training

During **generation**, vLLM handles all three steps internally. You send message lists, it applies the template, tokenizes, generates, returns text. You don't touch templates.

During **training**, the orchestrator has to reconstruct the exact same token sequence that vLLM generated — the full multi-turn conversation as one flat array of token IDs. It does this to:

1. Build the **loss mask** — 0 for prompt/system/user/special tokens, 1 for assistant completion tokens
2. Align **sampling logprobs** (from vLLM at generation time) with the correct positions
3. Compute **trainer logprobs** on the same sequence for the policy ratio

If the trainer's tokenizer applies the template differently from vLLM's — even one extra newline or a missing BOS token — the token IDs shift and the logprobs are misaligned. The loss is computed on wrong positions. No error, just bad gradients.

### Why You Don't Worry About This

Both vLLM and the trainer load the same model from HuggingFace → same `tokenizer_config.json` → same Jinja template → same tokenization. They agree automatically.

Your env code deals in message lists (step 1). The template (step 2) and tokenization (step 3) are handled by the infrastructure. The complaints about chat templates are from people who were stitching together mismatched pieces — different tokenizer versions, hardcoded format strings, models served with the wrong template. With a consistent model checkpoint loaded everywhere, the problem doesn't arise.

### The Multi-Turn Detail

For single-turn, the flat sequence is: prompt tokens → completion tokens. Simple mask.

For multi-turn (like email-to-cc-bcc), the flat sequence is:

```
[system] [user: email₁] [assistant: answer₁] [user: email₂] [assistant: answer₂] [user: email₃] [assistant: answer₃]
```

All one array. The model processes it left-to-right in one forward pass during training. The loss mask must be:

```
  0       0               1                0               1                0               1
```

The orchestrator figures out where each assistant turn starts and ends by applying the same template and finding the boundaries. This is where template correctness matters — if the boundaries are off by even one token, you're training on user tokens or skipping completion tokens.

During generation it's incremental — vLLM generates turn 1, the env injects the follow-up, vLLM generates turn 2 with the full history, etc. But during training, the entire conversation is one sequence processed in a single forward pass.
