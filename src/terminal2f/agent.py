from __future__ import annotations

from dataclasses import dataclass
from mistralai import Mistral


@dataclass
class Agent:
    """A simple AI agent"""
    client: Mistral
    model: str
    system_message: str
    tools: list
    max_tokens: int = 1024
    temperature: float = 0.7
    tool_choice: str = "auto"

    def act(self, messages: list, *, tools: list | None = None):
        """Call the model with given messages, return response"""
        tools = tools if tools is not None else self.tools
        return self.client.chat.complete(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=[t.schema for t in tools],
            tool_choice=self.tool_choice,  # type: ignore[arg-type]
            messages=[
                {"role": "system", "content": self.system_message},
                *messages,
            ]  # type: ignore[arg-type]
        )
