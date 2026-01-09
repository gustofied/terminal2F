import json
import pandas as pd

data = {
    "transaction_id": ["T1001", "T1002", "T1003", "T1004", "T1005"],
    "customer_id": ["C001", "C002", "C003", "C002", "C001"],
    "payment_amount": [125.50, 89.99, 120.00, 54.30, 210.20],
    "payment_date": ["2021-10-05", "2021-10-06", "2021-10-07", "2021-10-05", "2021-10-08"],
    "payment_status": ["Paid", "Unpaid", "Paid", "Paid", "Pending"],
}

df = pd.DataFrame(data)

def retrieve_payment_status(transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        status = df[df.transaction_id == transaction_id].payment_status.item()
        return json.dumps({"status": status})
    return json.dumps({"error": "transaction id not found."})

def retrieve_payment_date(transaction_id: str) -> str:
    if transaction_id in df.transaction_id.values:
        date = df[df.transaction_id == transaction_id].payment_date.item()
        return json.dumps({"date": date})
    return json.dumps({"error": "transaction id not found."})

names_to_functions = {
    "retrieve_payment_status": retrieve_payment_status,
    "retrieve_payment_date": retrieve_payment_date,
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
    },
]
