## TODO: t2f-trainer

### What This Is

`t2f-trainer` is our own RL trainer, forked from verifiers-rl. Lives at `external/t2f-trainer/`. The goal is to build our own trainer over time using verifiers-rl as the starting base.

### Done

- Extracted verifiers-rl out of the verifiers subtree into `external/t2f-trainer/`
- Renamed package to `t2f_trainer`, commands to `t2f-rl`, `t2f-train`, `t2f-vllm`
- All internal imports updated, standalone from upstream

### What We Have

The codebase is small and well-defined:

1. **`orchestrator.py`** - the multi-turn conversation loop. Model responds, environment gives new input, model responds again, score the whole trajectory. This is the core valuable piece.
2. **`client.py`** - vLLM weight sync. Pushes updated weights from training GPU to inference GPU via NCCL.
3. **`trainer.py`** - subclasses `transformers.Trainer` directly (not TRL). Runs GRPO with importance ratio clipping.
4. **`config.py`** - RLConfig with multi-turn fields.
5. **`server.py`** - custom vLLM server with weight sync endpoints.

### What We Learned

- verifiers-rl does NOT use TRL at all - it's pure `transformers.Trainer`
- The earlier plan to port into TRL's `rollout_func` is one option but not the only one
- Ludic exists as a purpose-built multi-turn/agentic RL library (custom trainer, no TRL)
- prime-rl is production-grade but not meant for hacking
- The real value is the orchestrator + weight sync, the trainer itself is fairly standard GRPO

### Directions

**Option A: Keep evolving t2f-trainer**
- Simplest path. We own it, it works, it's small enough to understand fully.
- Unpin vllm (currently locked to 0.10.x), make flash-attn optional
- Add KL penalty (training collapsed without it)
- Add more algorithms beyond GRPO
- Improve the TOML config support

**Option B: Study Ludic's architecture**
- Ludic is designed for exactly this - agentic multi-turn RL
- Clean separation: agents, environments, interaction protocols
- Modular credit assignment and loss functions
- Supports GRPO, SAPO, REINFORCE, SFT out of the box
- Could adopt ideas or patterns into t2f-trainer

**Option C: Contribute to prime-rl / upstream**
- If the changes are useful broadly, push them upstream
- Less control but more community

### Next Steps

1. Read through t2f-trainer source end to end - understand every file
2. Unpin vllm, test with latest
3. Make flash-attn optional (not everyone needs it)
4. Add KL penalty option to prevent training collapse
5. Read Ludic source for ideas on better multi-turn abstractions
6. Test on a GPU node with alphabet-sort
