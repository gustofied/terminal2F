import json
import logging

import control_tower
from tools import names_to_functions

log = logging.getLogger("app.runner")


def _get_state(agent) -> dict:
    state = agent.__dict__.get("_regular_runner_state")
    if state is None:
        state = {
            "turn_idx": 0,
            "messages": [{"role": "system", "content": agent.system_message}],
        }
        agent.__dict__["_regular_runner_state"] = state
    return state


def _run_tool_call(tool_call) -> tuple[str, dict]:
    function_name = tool_call.function.name
    function_params = json.loads(tool_call.function.arguments)

    # No guard rails on purpose. If the tool name is wrong, it should crash loud.
    result = names_to_functions[function_name](**function_params)
    return result, function_params


def run_agent(agent, user_message: str, max_turns: int = 10):
    state = _get_state(agent)
    messages = state["messages"]

    messages.append({"role": "user", "content": user_message})

    state["turn_idx"] += 1
    turn_idx = state["turn_idx"]
    control_tower.on_turn(turn_idx, user_message)

    prompt_tokens_max = 0

    response = agent.step(messages)
    prompt_tokens_max = max(prompt_tokens_max, response.usage.prompt_tokens)

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    tool_turns = 0
    while getattr(assistant_message, "tool_calls", None):
        tool_turns += 1
        if tool_turns > max_turns:
            raise RuntimeError("Max turns reached without a final answer.")

        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name

        function_result, function_params = _run_tool_call(tool_call)
        control_tower.on_tool_call(turn_idx, function_name, function_params)

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
    control_tower.on_assistant(turn_idx, final_text)
    control_tower.on_usage(turn_idx, prompt_tokens_max)

    return response
