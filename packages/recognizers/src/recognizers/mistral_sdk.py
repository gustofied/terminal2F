from dotenv import load_dotenv
import os
load_dotenv()
from mistralai import Mistral


class MistralSDKClient:
    def __init__(self, model="mistral-small-latest"):
        self.api_key = os.environ["MISTRAL_API_KEY"]
        self.model = model
        self.client = Mistral(api_key=self.api_key)
        self.tools = []
        self.name_to_tool = {}

    def register_tool(self, func, schema):
        self.name_to_tool[func.__name__] = func
        self.tools.append(schema)

    def chat(self, user_message, tool_choice="any", parallel_tool_calls=False):
        messages = [{"role": "user", "content": user_message}]
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,  # ty:ignore[invalid-argument-type]
            tools=self.tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        return response


# --- tools ---

def count(start_number: int, end_number: int):
    for x in range(start_number, end_number + 1):
        print(f"count: {x}")


count_schema = {
    "type": "function",
    "function": {
        "name": "count",
        "description": "Count between two numbers",
        "parameters": {
            "properties": {
                "start_number": {
                    "type": "integer",
                    "description": "Start counting from this number",
                },
                "end_number": {
                    "type": "integer",
                    "description": "End the count using this number",
                }
            },
            "required": ["start_number", "end_number"]
        }
    }
}


if __name__ == "__main__":
    sdk = MistralSDKClient()
    sdk.register_tool(count, count_schema)
    response = sdk.chat("Can you Count from 10 to 100, use a tool budy")
    print(response)
