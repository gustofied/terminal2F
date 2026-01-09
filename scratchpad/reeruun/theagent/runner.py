import json
from tools import names_to_functions
import logging
import rerun as rr

import logging
log = logging.getLogger("app.runner")

def context_char_len(agent) -> int:
    total = 0
    for m in agent.messages:
        if isinstance(m, dict):
            total += len(m.get("content", ""))
        else:
            # AssistantMessage or similar SDK object
            total += len(getattr(m, "content", "") or "")
    return total

def run_agent(agent, user_message: str, max_turns: int = 10):


    turn_idx = 0
    log.debug(f"context char length before turn = {context_char_len(agent)}")
    agent.messages.append({"role": "user", "content": user_message})


    turn_idx += 1
    rr.set_time("turn", sequence=turn_idx)
    rr.log(
        "agent/conversation",
        rr.TextLog(f"user: {user_message}"),
    )

    response = agent.step()
    assistant_message = response.choices[0].message
    
    

    turns = 0
    while getattr(assistant_message, "tool_calls", None):
        turn_idx += 1
        turns += 1
        if turns > max_turns:
            raise RuntimeError("Max turns reached without a final answer.")

        tool_call = assistant_message.tool_calls[0]

        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)
        rr.log(
            "agent/tool_calls",
            rr.TextLog(f"{function_name}({function_params})"),
        )


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

    log.debug(f"after the turn ={context_char_len(agent)}")

    size = context_char_len(agent)
    log.debug(size)

    rr.log("context", rr.Points2D([1, 1.0], radii=size, colors=[255, 200, 10]))

    

    return response
