# t2f-trainer

Fork of [verifiers-rl](https://github.com/PrimeIntellect-ai/verifiers/tree/main/packages/verifiers-rl), extracted into its own package. Same core (RLTrainer, orchestrator, vLLM weight sync) but renamed so it can be modified independently.

```bash
uv pip install -e ./external/t2f-trainer
```

Commands: `t2f-rl`, `t2f-train`, `t2f-vllm`

Import: `from t2f_trainer.rl.trainer import RLTrainer, RLConfig`

## Changes from original verifiers-rl

**`get_model_and_tokenizer` import** — The original calls `vf.get_model_and_tokenizer()` from the verifiers package, which gates it behind a `verifiers-rl` install check. Since t2f-trainer has its own copy of this function in `utils.py`, the import was changed to use the local version. Without this, installing plain `verifiers` (without the rl extra) raises `AttributeError: To use verifiers.get_model_and_tokenizer, install as verifiers-rl`.

```python
# before (trainer.py)
model, processing_class = vf.get_model_and_tokenizer(model, use_liger=args.use_liger)

# after
from t2f_trainer.rl.trainer.utils import get_model_and_tokenizer
model, processing_class = get_model_and_tokenizer(model, use_liger=args.use_liger)
```

**`flash-attn` made optional** — `flash-attn` compiles CUDA kernels from source against your exact torch + CUDA versions (~15 min). For small models with `enforce_eager = true`, it is not needed. Commented out in `pyproject.toml` so install is fast (just wheels, no compilation). To use flash-attn, uncomment it and install with `--no-build-isolation` after installing torch.

```toml
# pyproject.toml
dependencies = [
    ...
    # "flash-attn>=2.8.3",  # optional: compiles from source, use enforce_eager without it
]
```

**`ClientConfig` instead of `AsyncOpenAI`** — verifiers 0.1.11+ routes all clients through `resolve_client()`, which only accepts `Client` or `ClientConfig`, not raw `AsyncOpenAI`. The orchestrator was creating an `AsyncOpenAI` directly and passing it to `env.generate()`. Changed to pass a `ClientConfig` instead, which verifiers constructs the client from internally. The API key is set as an env var (`T2F_VLLM_API_KEY`) since `ClientConfig.api_key_var` expects an env var name, not a literal value.

```python
# before (orchestrator.py)
self.client = AsyncOpenAI(base_url=..., api_key="EMPTY", ...)

# after
os.environ.setdefault("T2F_VLLM_API_KEY", self.client_api_key)
self.client = ClientConfig(
    client_type="openai_chat_completions",
    api_base_url=self.client_base_url,
    api_key_var="T2F_VLLM_API_KEY",
    ...
)
```
