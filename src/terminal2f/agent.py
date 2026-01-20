import os
import uuid
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral

from .agent_profiles import AgentProfile, get_profile

load_dotenv()


class Agent:
    def __init__(
        self,
        tools_installed,
        *,
        profile: AgentProfile | None = None,
        model: str | None = None,
        name: str = "agent",
        instance_id: str | None = None,
    ):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("Missing MISTRAL_API_KEY in environment")

        self.client = Mistral(api_key=api_key)

        self.profile: AgentProfile = profile or get_profile("default")
        self.model = model or self.profile.model.model

        self.tools_installed = tools_installed or []

        model_info = self.client.models.retrieve(model_id=self.model)
        self.max_context_length = model_info.max_context_length

        self.name = name
        self.instance_id = instance_id or uuid.uuid4().hex[:8]
        self.system_message = self.profile.render_system_message(cwd=os.getcwd())

    def step(
        self,
        messages,
        *,
        tools_exposed: list[dict] | None = None,
    ):
        tools_exposed = tools_exposed or []
        profile = self.profile

        kwargs: dict[str, Any] = dict(
            model=profile.model.model or self.model,
            messages=messages,
            temperature=float(profile.model.temperature),
            max_tokens=int(profile.model.max_tokens),
        )

        if tools_exposed:
            kwargs["tools"] = tools_exposed
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = bool(profile.model.parallel_tool_calls)

        return self.client.chat.complete(**kwargs)
