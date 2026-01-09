import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

import logging
log = logging.getLogger("app.agent")

class Agent:
    def __init__(self, tools, model: str = "mistral-medium-latest"):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.model = model
        self.tools = tools
        self.system_message = (
            "You are a helpful assistant that breaks down problems into steps and solves them systematically. "
            "Write max 3 sentences. "
            "Use tools only for payment transaction questions."
        )
        self.messages = [{"role": "system", "content": self.system_message}]
        self.turn_idx = 0

    def step(self):
        """
        One LLM step. Runner decides what to do with tool_calls.
        """
        response = self.client.chat.complete(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.1,
            max_tokens=1024,
        )

        assistant_message = response.choices[0].message
        self.messages.append(assistant_message)
        return response
