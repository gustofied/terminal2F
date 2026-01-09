import json
from tools import names_to_functions

def run_agent(agent, user_message: str, max_turns: int = 10):
    """
    Pattern B runner:
    - add user message
    - call agent.step()
    - if tool_calls: execute tool, append tool result, step again
    - returns FINAL response object
    """

    agent.messages.append({"role": "user", "content": user_message})

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
        print(f"[tool] {function_name}({function_params})")

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

    return response
