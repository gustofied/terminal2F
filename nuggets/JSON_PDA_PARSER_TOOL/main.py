from dotenv import load_dotenv
import os
load_dotenv()
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-latest"

def count(start_number: int, end_number: int):
    for x in range(start_number, end_number + 1):
        print(f"count: {x}")

# count(10, 20)

# name_to_tool = {"count": count}
# name_to_tool["count"](2,50)
# name_to_tool.get("count", 0)(2, 20)

schemaen = {
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

client = Mistral(api_key=api_key)

messages = []
messages.append({"role": "user", "content": "Can you Count from 10 to 100, use a tool budy"})

response = client.chat.complete(
    model = model, # The model we want to use
    messages = messages, # The message history, in this example we have a system (optional) + user query.
    tools = [schemaen], # The tools specifications
    tool_choice = "any",
    parallel_tool_calls = False,
    response_format = {
          "type": "json_object",
      }
)

print(response)