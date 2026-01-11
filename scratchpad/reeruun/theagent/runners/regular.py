import json
import logging

import control_tower
from tools import names_to_functions

log = logging.getLogger("app.runner")


def run_agent(agent, user_message: str, max_turns: int = 10):
    state = agent.__dict__.setdefault(
        "_regular_runner_state",
        {
            "agent_key": hex(id(agent))[2:],
            "turn_idx": 0,
            "messages": [{"role": "system", "content": agent.system_message}],
        },
    )

    agent_key = state["agent_key"]
    messages = state["messages"]

    messages.append({"role": "user", "content": user_message})

    state["turn_idx"] += 1
    turn_idx = state["turn_idx"]
    control_tower.on_turn(agent_key, turn_idx, user_message)

    prompt_tokens_max = 0

    response = agent.step(messages)
    prompt_tokens_max = max(prompt_tokens_max, response.usage.prompt_tokens)

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    turns = 0
    while getattr(assistant_message, "tool_calls", None):
        turns += 1
        if turns > max_turns:
            raise RuntimeError("Max turns reached without a final answer.")

        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)

        control_tower.on_tool_call(agent_key, turn_idx, function_name, function_params)

        function_result = names_to_functions[function_name](**function_params)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_result,
            }
        )

        response = agent.step(messages)
        prompt_tokens_max = max(prompt_tokens_max, response.usage.prompt_tokens)

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

    final_text = getattr(assistant_message, "content", "") or ""
    control_tower.on_assistant(agent_key, turn_idx, final_text)
    control_tower.on_usage(agent_key, turn_idx, prompt_tokens_max)

    return response
