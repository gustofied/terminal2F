import os
import logging
import uuid
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()
log = logging.getLogger("app.agent")


class Agent:
    def __init__(
        self,
        tools,
        model: str = "ministral-3b-2512",
        name: str = "agent",
        instance_id: str | None = None,
    ):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.model = model
        self.tools = tools

        self.name = name
        # Unique id, unsure about this one
        self.instance_id = instance_id or uuid.uuid4().hex[:8]

        self.system_message = (
            "You are a helpful agent that breaks down problems into steps and solves them systematically. "
            "Write max 3 sentences. "
            "Use tools only for payment transaction questions."
        )

    def step(self, messages):
        return self.client.chat.complete(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.1,
            max_tokens=1024,
        )
