import json
from tools import names_to_functions
import control_tower


def _prompt_tokens(response) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get("prompt_tokens")
    return getattr(usage, "prompt_tokens", None)


class RegularRunner:
    """
    Regular runner.
    This is your current behavior, but runner owns memory.
    Runner builds the messages list and passes it into agent.step(messages).
    """

    def __init__(self, *, agent, max_turns: int = 10):
        self.agent = agent
        self.max_turns = max_turns

        self.turn_idx = 0
        self.messages = [{"role": "system", "content": self.agent.system_message}]

    def run(self, user_message: str):
        self.turn_idx += 1
        control_tower.on_turn(self.turn_idx, user_message)

        self.messages.append({"role": "user", "content": user_message})

        prompt_tokens_max = 0

        response = self.agent.step(self.messages)
        p = _prompt_tokens(response)
        if isinstance(p, int):
            prompt_tokens_max = max(prompt_tokens_max, p)

        assistant_message = response.choices[0].message
        self.messages.append(assistant_message)

        turns = 0
        while getattr(assistant_message, "tool_calls", None):
            turns += 1
            if turns > self.max_turns:
                raise RuntimeError("Max turns reached without a final answer.")

            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)

            control_tower.on_tool_call(self.turn_idx, function_name, function_params)

            tool_function = names_to_functions.get(function_name)
            if tool_function is None:
                function_result = json.dumps({"error": f"Unknown tool: {function_name}"})
            else:
                function_result = tool_function(**function_params)

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_result,
                }
            )

            response = self.agent.step(self.messages)
            p = _prompt_tokens(response)
            if isinstance(p, int):
                prompt_tokens_max = max(prompt_tokens_max, p)

            assistant_message = response.choices[0].message
            self.messages.append(assistant_message)

        final_text = getattr(assistant_message, "content", "") or ""
        control_tower.on_assistant(self.turn_idx, final_text)

        if prompt_tokens_max:
            control_tower.on_usage(self.turn_idx, prompt_tokens_max)

        return response
