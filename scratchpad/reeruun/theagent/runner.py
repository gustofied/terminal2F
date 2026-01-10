import json
import logging
from tools import names_to_functions
import control_tower as control_tower

log = logging.getLogger("app.runner")

def context_char_len(agent) -> int:
    total = 0
    for m in agent.messages:
        if isinstance(m, dict):
            total += len(m.get("content", "") or "")
        else:
            total += len(getattr(m, "content", "") or "")
    return total

def run_agent(agent, user_message: str, max_turns: int = 10):
    log.debug("context char length before turn = %s", context_char_len(agent))

    agent.messages.append({"role": "user", "content": user_message})

    agent.turn_idx += 1
    control_tower.on_turn(agent.turn_idx, user_message)

    response = agent.step()
    assistant_message = response.choices[0].message

    turns = 0
    while getattr(assistant_message, "tool_calls", None):
        turns += 1
        if turns > max_turns:
            raise RuntimeError("Max turns reached without a final answer.")

        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)

        control_tower.on_tool_call(agent.turn_idx, function_name, function_params)

        tool_function = names_to_functions.get(function_name)
        if tool_function is None:
            function_result = json.dumps({"error": f"Unknown tool: {function_name}"})
        else:
            function_result = tool_function(**function_params)

        agent.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_result,
        })

        response = agent.step()
        assistant_message = response.choices[0].message

    final_text = getattr(assistant_message, "content", "") or ""
    control_tower.on_assistant(agent.turn_idx, final_text)

    after = context_char_len(agent)
    log.debug("context char length after turn = %s", after)
    control_tower.on_context(agent.turn_idx, after)

    return response
