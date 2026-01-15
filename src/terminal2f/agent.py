import os
import uuid
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral
from .env import Env, get_env

load_dotenv()


class Agent:
    def __init__(
        self,
        tools_installed,
        *,
        env: Env | None = None,
        model: str | None = None,
        name: str = "agent",
        instance_id: str | None = None,
    ):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY in environment")

        self.client = Mistral(api_key=api_key)

        self.env = env or get_env("default")
        self.model = model or self.env.agent.model

        self.tools_installed = tools_installed or []

        model_info = self.client.models.retrieve(model_id=self.model)
        self.max_context_length = model_info.max_context_length

        self.name = name
        self.instance_id = instance_id or uuid.uuid4().hex[:8]
        self.system_message = self.env.render_system_message(cwd=os.getcwd())

    def step(self, messages, *, tools_exposed: list[dict] | None = None, env: Env | None = None):
        env = env or self.env
        tools_exposed = tools_exposed or []

        kwargs: dict[str, Any] = dict(
            model=env.agent.model or self.model,
            messages=messages,
            temperature=float(env.agent.temperature),
            max_tokens=int(env.agent.max_tokens),
        )

        if tools_exposed:
            kwargs["tools"] = tools_exposed
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = bool(env.agent.parallel_tool_calls)

        return self.client.chat.complete(**kwargs)
