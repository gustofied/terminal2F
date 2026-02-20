# t2f-trainer

Fork of [verifiers-rl](https://github.com/PrimeIntellect-ai/verifiers/tree/main/packages/verifiers-rl) that I've been hacking on. Same core (RLTrainer, orchestrator, vLLM weight sync) but renamed and extracted so I can modify it without creating a mess.

based on verifiers-rl, extended over time with, up to date.

```bash
uv pip install -e ./external/t2f-trainer
```

Commands: `t2f-rl`, `t2f-train`, `t2f-vllm`

Import: `from t2f_trainer.rl.trainer import RLTrainer, RLConfig`
