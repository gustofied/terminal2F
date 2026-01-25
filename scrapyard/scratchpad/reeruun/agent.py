import os
from mistralai import Mistral
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "ministral-3b-2512"

# tools for the agent / functoin calling
# Assuming we have the following data
data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}
df = pd.DataFrame(data)

def retrieve_payment_status(transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'transaction id not found.'})

names_to_functions = {
    'retrieve_payment_status': retrieve_payment_status,
    'retrieve_payment_date': retrieve_payment_date,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string", "description": "The transaction id."}
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string", "description": "The transaction id."}
                },
                "required": ["transaction_id"],
            },
        },
    }
]

class Agent:
    def __init__(self):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.tools = tools
        self.system_message = (
            "You are a helpful assistant that breaks down problems into steps and solves them systematically. "
            "Write max 3 sentences. "
            "Use tools only for payment transaction questions."
        )
        self.messages = [{"role": "system", "content": self.system_message}]

    def chat(self, message: str):
        # 1) add user message
        self.messages.append({"role": "user", "content": message})

        response = self.client.chat.complete(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.1,
            max_tokens=1024,
        )

        msg = response.choices[0].message
        self.messages.append(msg)

        while getattr(msg, "tool_calls", None):
            tool_call = msg.tool_calls[0]  

            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)
            print(f"[tool] {function_name}({function_params})")


            tool_function = names_to_functions.get(function_name)

            if tool_function is None:
                function_result = json.dumps({"error": f"Unknown tool: {function_name}"})
            else:
                function_result = tool_function(**function_params)

            # send tool result back to model
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_result,  # JSON string
            })

            response = self.client.chat.complete(
                model=self.model,
                messages=self.messages,
                temperature=0.1,
                max_tokens=1024,
            )
            msg = response.choices[0].message
            self.messages.append(msg)

        return response


if __name__ == "__main__":
    agent = Agent()

    response = agent.chat("I have 4 apples. How many do you have?")
    print(response.choices[0].message.content)
    print("- - - - - - - - - - - -")

    response = agent.chat("I ate 1 apple. How many are left?")
    print(response.choices[0].message.content)
    print("- - - - - - - - - - - -")

    response = agent.chat("What is 157.09 * 493.89?")
    print(response.choices[0].message.content)
    print("- - - - - - - - - - - -")

    response = agent.chat("What's the status of my transaction T1001?")
    print(response.choices[0].message.content)
