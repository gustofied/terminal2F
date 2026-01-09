import os
from mistralai import Mistral
from dotenv import load_dotenv
import pandas as pd
import json

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-medium-latest"

# tools for the agent / functoin calling

# Assuming we have the following data
data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}

# Create DataFrame
df = pd.DataFrame(data)

def retrieve_payment_status(transaction_id: str) -> str:
    "Get payment status of a transaction"
    if transaction_id in df.transaction_id.values:
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(transaction_id: str) -> str:
    "Get payment date of a transaction"
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
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
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
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]

class Agent:
    """A simple AI agent that can answer questions"""

    def __init__(self):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.tools = tools
        self.system_message = (
            "You are a helpful assistant that breaks down problems into steps and solves them systematically. "
            "Write max 3 sentences. "
            "You can use the following tools to help answer the user's questions related to payment transactions."
        )
        self.messages = [{"role": "system", "content": self.system_message}]

    def chat(self, message: str):
        # Store user input
        self.messages.append({"role": "user", "content": message})

        response = self.client.chat.complete(
            model=self.model,
            max_tokens=1024,
            temperature=0.1,
            tools = self.tools, # The tools specifications
            tool_choice = "auto",
            parallel_tool_calls = False,
            messages=self.messages,
        )

        self.messages.append(response.choices[0].message)


        return response
    
agent = Agent()

response = agent.chat("I have 4 apples. How many do you have?")
print(response.choices[0].message.content)

print("- - - - - - - - - - - -")

response = agent.chat("I ate 1 apple. How many are left?")
print(response.choices[0].message.content)

print("- - - - - - - - - - - -")

response = agent.chat("What is 157.09 * 493.89?")

print(response.choices[0].message.content)


response = agent.chat("What's the status of my transaction T1001?")
print(response.choices[0].message.content)
