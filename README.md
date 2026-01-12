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

#### How it currently work, I'm inside theagent folde btw

agent.py only does model calls, runner.py the loop + tool execution, and control_tower.py does all Rerun logging.

##### Expirments

setup, a single Rerun recording is one experiment. An experiment can contain multiple agents running side by side, and each agent instance produces its own logs and metrics while sharing the same experiment context.

Within an experiment, you create episodes by clearing the conversational state. A clear does not rewind time. It resets the agent’s message history so context length drops naturally, and it logs a clear event so the episode boundary is explicit and easy to spot later. Because the turn counter is monotonic for the entire experiment, nothing gets overwritten in Rerun and you can reliably compare behavior across episodes, across agents, and across different runner implementations inside the same experiment.

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

Later add per-turn "step" indexing (1.1, 1.2, 1.3) for internal tool calls also when talking to sub-agents. Keep turn_idx as the user interaction counter. Add step_idx inside this turn and pass it into control_tower hooks. That will let Rerun show ordering within a single user turn.

##### Tokens

prompt_tokens is the size of what is sent into the model for this call.
might be interested to look at response.usage.total_tokens or
response.usage.completion_token
the model also needs room for the output so I can't just use prompt tokens like now..

##### Jinja Templates

introduce jinja templates in the future, that's going to be poerfull yeah
