from dotenv import load_dotenv
import os
import requests
import json
from pathlib import Path

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
url = "https://api.mistral.ai/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Bearer {api_key}",
}


# --- tools ---

def lookup_order(order_id: str):
    return {"order_id": order_id, "status": "shipped", "eta": "2026-02-15"}

def lookup_user(username: str):
    return {"username": username, "name": "Anders Eriksen", "email": "anders.eriksen@mail.no"}


name_to_tool = {"lookup_order": lookup_order, "lookup_user": lookup_user}

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up order status by order ID",
            "parameters": {
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "Look up user info by username",
            "parameters": {
                "properties": {
                    "username": {"type": "string", "description": "The username"}
                },
                "required": ["username"]
            }
        }
    }
]

MODELS = {
    "small": "mistral-small-latest",
    "reasoning": "magistral-small-latest",
}

def post(messages, model="small", stream=True, tool_choice="auto"):
    payload = {
        "model": MODELS[model],
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "stream": stream,
    }
    return requests.post(url, headers=headers, json=payload, stream=stream)

system_message = "Skriv på norsk. Tenk steg for steg. Bruk ett verktøy om gangen, forklar hva du gjør mellom hvert kall."
messages = [{"role": "system", "content": system_message},]

class LOOP:
    def __init__(self, user_input: str, messages: list, *, tools: list | None = None, max_turns=10):
        self.user_input = user_input
        self.messages = messages
        self.tools = tools
        self.max_turns = max_turns

    def __call__(self):
        self.messages.append({"role": "user", "content": self.user_input})
        data_dir = Path(__file__).parent / "data"

        for turn in range(self.max_turns):
            response = post(self.messages)
            last_chunk = None
            with open(data_dir / f"turn_{turn}.json", "w") as f:
                for line in response.iter_lines():
                    if line:
                        raw = line.decode("utf-8")
                        f.write(raw + "\n")
                        print(raw)
                        if raw.startswith("data: ") and raw != "data: [DONE]":
                            last_chunk = json.loads(raw[6:])

            # check if the last chunk had tool calls
            if not last_chunk:
                break
            finish = last_chunk["choices"][0].get("finish_reason")
            if finish != "tool_calls":
                break

            # extract tool calls from last chunk, execute, append results
            delta = last_chunk["choices"][0]["delta"]
            self.messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                    for tc in delta["tool_calls"]
                ],
            })
            for tc in delta["tool_calls"]:
                result = name_to_tool[tc["function"]["name"]](**json.loads(tc["function"]["arguments"]))
                self.messages.append({
                    "role": "tool",
                    "name": tc["function"]["name"],
                    "content": str(result),
                    "tool_call_id": tc["id"],
                })


if __name__ == "__main__":
    email = """Hi support, my name is Anders Eriksen (user: anders.eriksen@mail.no) and I have an issue with order #ORD-90421. The package never arrived. Please help."""
    prompt = f"Use your tools to look up the user and the order from this email, then respond with a summary.\n\n{email}"
    loop = LOOP(prompt, messages)
    loop()
