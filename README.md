<div align="center">
<img src="assets/terminaldeuxbanny.png" alt="terminal2F Banner" width="600">
<h1>terminal2F</h1>
</div>
<div align="center">
<b>A way to observe agents and agent systems</b>
</div>
<br>
<div align="center">
<i>Monitor • Trace • Visualize LLM calls.</i>
</div>

---

### Notes

#### TO-DOS

##### QueueHandler/QueueListener + Rerun can get weird on quit

When putting logging behind a queue, the actual send to Rerun happens on the listener thread, and Rerun's flush guarantees are basically "for the calling thread" (other threads can still have stuff in flight). So it can look like the queue handler kinda fucks with the Rerun logs at shutdown (last messages missing / not fully flushed).

**Current approach:** Keeping it simple (no queue) for now.

**Future consideration:** If/when the queue is re-added, it likely needs an explicit shutdown order:

1. Stop listener
2. Flush/disconnect

##### Token Prediction

Add token prediction so you I see how much context a request will take before sending it. Pretty useful for staying under limits and knowing when to trim history. Will use the Mistral's tokenizer from mistral-common since we're on Mistral models. Viz this too with two scalars, prediction and actual..

##### Step inside Turns

Later add per-turn "step" indexing (1.1, 1.2, 1.3) for internal tool calls also when talking to sub-agents. # Keep turn_idx as the user interaction counter. Add step_idx inside this turn and pass it into control_tower hooks. That will let Rerun show ordering within a single user turn.
