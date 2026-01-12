import json
import logging

from .. import control_tower
from ..tools import names_to_functions

log = logging.getLogger("app.runner")


def run_agent(agent, user_message: str, max_turns: int = 10, ui=None):
    state = agent.__dict__.setdefault(
        "_regular_runner_state",
        {
            "instance_id": getattr(agent, "instance_id", hex(id(agent))[2:]),
            "agent_name": getattr(agent, "name", "agent"),
            "turn_idx": 0,
            "messages": [{"role": "system", "content": agent.system_message}],
        },
    )

    instance_id = state["instance_id"]
    agent_name = state["agent_name"]
    messages = state["messages"]

    messages.append({"role": "user", "content": user_message})

    state["turn_idx"] += 1
    turn_idx = state["turn_idx"]
    control_tower.on_turn(agent_name, instance_id, turn_idx, user_message)

    context_window = 0

    def _ui_call(method_name: str, *args):
        if not ui:
            return
        fn = getattr(ui, method_name, None)
        if callable(fn):
            fn(*args)

    response = agent.step(messages)
    context_window = max(context_window, response.usage.prompt_tokens)

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    # Print assistant text immediately (if any)
    text = getattr(assistant_message, "content", "") or ""
    if text:
        _ui_call("on_assistant_text", text)

    turns = 0
    while getattr(assistant_message, "tool_calls", None):
        turns += 1
        if turns > max_turns:
            raise RuntimeError("Max turns reached without a final answer.")

        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)

        control_tower.on_tool_call(agent_name, instance_id, turn_idx, function_name, function_params)
        _ui_call("on_tool_call", function_name, function_params)

        function_result = names_to_functions[function_name](**function_params)
        _ui_call("on_tool_result", function_name, function_result)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_result,
            }
        )

        response = agent.step(messages)
        context_window = max(context_window, response.usage.prompt_tokens)

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        text = getattr(assistant_message, "content", "") or ""
        if text:
            _ui_call("on_assistant_text", text)

    final_text = getattr(assistant_message, "content", "") or ""
    control_tower.on_assistant(agent_name, instance_id, turn_idx, final_text)

    control_tower.on_usage(
        agent_name,
        instance_id,
        turn_idx,
        context_window,
        context_limit=agent.max_context_length,
    )

    return response
