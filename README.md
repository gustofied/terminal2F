<div align="center">
<img src="assets/terminaldeuxbanny.png" alt="terminal2F Banner" width="600">
<h1>terminal2F <i>(WIP)</i></h1>
<b>A research platform for agents and agent systems</b>
<br><br>
<i>Monitor • Evaluate • Train</i>

</div>

<br>

<p align="center">
<img src="assets/demo_t2f.gif" alt="terminal2F Demo" width="800">
</p>

---

## How To Run

```bash
# Interactive chat (default loop runner)
t2f chat

# Chat with a specific automaton/runner
t2f chat --automaton fsm    # or pda, tm, lba

# Start Rerun observability server
t2f serve record             # run experiment, write .rrd files
t2f serve load <run_id>      # load existing run into viewer
t2f serve live               # stream to viewer in real-time (stub)
```

Requires `MISTRAL_API_KEY` in your `.env`. The `t2f` command is available after installing the package. Switch automaton during chat with `/automaton <name>`. Exit with `/q` or `quit`.

---

## What is this

---

## Research Harness

#### Engine

`run.py` is the engine behind it all. It handles experiment setup, generates run IDs, creates Rerun recording streams, and manages the per-episode context. You define your policies and environments, the engine iterates over episodes, records everything to `.rrd` files, and logs metrics to the catalog. The experiment runs as a context manager, each episode gets its own recording, and everything is timestamped and traceable.

---

## Primitives

#### Agent

The Agent is a single chat completion call. It holds config (model, temperature, system prompt, tools) and exposes one method: act. No memory, no loop, no state. That all lives in the automaton.

#### Automata

The runners that drive the agent. LOOP is your typical agent loop implementation. FSM is an explicit state machine with bounded context (k=3). PDA is stack-top driven with full history. LBA extends PDA with a bounded read/write scratchpad. TM extends PDA with unbounded read/write memory. Same "agent", different runner.

#### Systems

A Clock is the execution environment for N agents on a shared clock. One root agent owns the clock, sub-agents are spawned into it. All agents in a clock share an object store (the shared artifact space) and tick on the same clock. The clock decides when agents run, what they can see, and when they're done. Inspired by [State Machines for Multi-Agent](https://eriksfunhouse.com/writings/state_machines_for_multi_agent_part_1/) and [P2Engine](https://github.com/gustofied/P2Engine).

#### Memory

There are three layers of memory: raw messages (for the typical agent loop), a typed interaction stack (for the state machine runners), and an object store (long-term artifacts, TM-level). What gets used and how is up to the automaton. The memory architecture is what determines the agent's computational power: bounded context gives you an FSM, a stack gives you a PDA, read/write memory gives you a TM.

#### Environments

An environment is a task that the agent tries to solve. It gives the agent an observation (the problem), the agent produces an answer, and the environment scores it (reward). Reset starts a new task, step takes the agent's answer and returns the next observation, a reward, and whether it's done.

#### Tools

Currently tools are implemented in the standard way per [Mistral function calling](https://docs.mistral.ai/capabilities/function_calling).

---

## Data Model

terminal2F uses Rerun's [data platform](https://rerun.io/docs/concepts/query-and-transform/catalog-object-model), a DataFusion-based platform where data is served via the redap protocol (Rerun Data Protocol). The top level is the catalog, which maps to our experiments.

In the catalog there are two types of entries: **datasets** and **tables**. Datasets hold the recordings from experiments. Tables hold metrics (runs metadata, episode scores, etc).

The dataset is not storage. It is a workspace, more like a viewer of data. On a new run we DELETE + RECREATE the dataset to get a clean slate. The actual data lives in `.rrd` files on disk. The dataset just tells the viewer what to show.

```
┌─────────────────────────────────────┐
│  Experiment (stable name)           │
│  EXPERIMENT_FAMILY/VERSION          │
└──────────────┬──────────────────────┘
               │
               │  many
               v
┌─────────────────────────────────────┐
│  runs (table)                       │
│  run_id (ULID), started_at, ended_at│
└──────────────┬──────────────────────┘
               │
               │  many rows per run
               v
┌─────────────────────────────────────┐
│  episodes (table)                   │
│  run_id, episode_id, layer (variant)│
│  total_return, steps, done          │
└──────────────┬──────────────────────┘
               │
               │  points to immutable artifacts
               v
┌─────────────────────────────────────┐
│  Artifacts (.rrd files on disk)     │
│                                     │
│  logs/storage/recordings/           │
│    <exp>/<version>/runs/            │
│      <run_id>/                      │
│        <layer>/                     │
│          <episode_id>.rrd           │
└─────────────────────────────────────┘


┌─────────────────────────────────────────────────┐
│  Dataset: EXPERIMENT (workspace / viewer)        │
│                                                  │
│  segment_id = episode_id                         │
│  layer      = variant (loop/fsm/pda/...)         │
│                                                  │
│  On new run: DELETE + RECREATE to start clean.   │
│                                                  │
│  Problem (segment = episode_1)                   │
│   ├── loop  (layer)  -> episode_1.rrd            │
│   ├── fsm   (layer)  -> episode_1.rrd            │
│   └── pda   (layer)  -> episode_1.rrd            │
│                                                  │
│  Problem (segment = episode_2)                   │
│   ├── loop  (layer)  -> episode_2.rrd            │
│   └── ...                                        │
└─────────────────────────────────────────────────┘
```

---
