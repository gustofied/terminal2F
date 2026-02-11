from __future__ import annotations

from dataclasses import dataclass


# --- Interaction types ---
# Typed entries for the interaction stack. Each has a to_message() that returns
# the API dict (or None if not renderable). The stack is the single source of truth.

@dataclass
class UserMessage:
    content: str
    def to_message(self) -> dict:
        return {"role": "user", "content": self.content}

@dataclass
class AssistantMessage:
    content: str
    tool_calls: list | None = None
    def to_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

@dataclass
class ToolCall:
    """Control flow marker — not rendered to API."""
    name: str
    args: dict
    tool_call_id: str
    def to_message(self) -> dict | None:
        return None

@dataclass
class ToolResult:
    name: str
    output: str
    tool_call_id: str
    def to_message(self) -> dict:
        return {"role": "tool", "name": self.name, "content": self.output, "tool_call_id": self.tool_call_id}

@dataclass
class AgentCall:
    """Parent asked a sub-agent to do work."""
    agent_name: str
    instruction: str
    def to_message(self) -> dict | None:
        return None  # control flow only

@dataclass
class AgentResult:
    """Sub-agent finished, result returned to parent."""
    agent_name: str
    result: str
    def to_message(self) -> dict:
        return {"role": "user", "content": self.result}  # fed back as user msg to parent


@dataclass
class Finished:
    """Terminal marker — never popped, signals completion."""
    result: str
    def to_message(self) -> dict | None:
        return None

# move later
def render_context(stack: list, *, k: int | None = None) -> list[dict]:
    """Build API messages from the interaction stack.
    k=N for bounded window (FSM), k=None for full history (PDA/LBA/TM)."""
    entries = stack if k is None else stack[-k:]
    return [msg for entry in entries if (msg := entry.to_message()) is not None]
