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

        model_info = self.client.models.retrieve(model_id=self.model)
        self.max_context_length = model_info.max_context_length

        self.name = name
        self.instance_id = instance_id or uuid.uuid4().hex[:8]

        self.system_message = f"Concise assistand and or a coding assistant. cwd: {os.getcwd()}"

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
