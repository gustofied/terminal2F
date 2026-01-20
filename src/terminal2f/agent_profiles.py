from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet

from .tools.toolsets import ALL_TOOLS, NO_TOOLS, PAYMENTS

ToolSchema = dict[str, Any]


def tool_name(schema: ToolSchema) -> str:
    return schema["function"]["name"]


@dataclass(frozen=True)
class ToolPolicy:
    allowed: FrozenSet[str] = frozenset()

    def allows(self, name: str) -> bool:
        return bool(name) and ("*" in self.allowed or name in self.allowed)


@dataclass(frozen=True)
class ModelConfig:
    model: str = "ministral-3b-2512"
    temperature: float = 0.1
    max_tokens: int = 1024
    parallel_tool_calls: bool = False


@dataclass(frozen=True)
class AgentProfile:
    name: str
    description: str = ""
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    ctx_budget: int | None = 5000
    max_tool_turns: int = 10
    model: ModelConfig = field(default_factory=ModelConfig)
    system_message: str = "Concise coding assistant. profile={profile} cwd={cwd}"

    def render_system_message(self, *, cwd: str) -> str:
        return self.system_message.format(profile=self.name, cwd=cwd)


def compile_tools(
    *,
    profile: AgentProfile,
    installed_tools: list[ToolSchema],
    requested_tools: list[ToolSchema] | None,
) -> tuple[set[str], list[ToolSchema]]:
    installed = installed_tools or []
    basis = installed if requested_tools is None else (requested_tools or [])

    installed_names = {tool_name(t) for t in installed}

    exposed: list[ToolSchema] = []
    for t in basis:
        n = tool_name(t)
        if n in installed_names and profile.tool_policy.allows(n):
            exposed.append(t)

    return {tool_name(t) for t in exposed}, exposed


PROFILES: dict[str, AgentProfile] = {}

PROFILES["default"] = AgentProfile(
    name="default",
    description="payments tools only",
    tool_policy=ToolPolicy(allowed=PAYMENTS),
    ctx_budget=5000,
    max_tool_turns=10,
    model=ModelConfig(
        model="ministral-3b-2512",
        temperature=0.1,
        max_tokens=1024,
        parallel_tool_calls=False,
    ),
    system_message="Concise coding assistant. profile={profile} cwd={cwd}",
)

PROFILES["chat_safe"] = AgentProfile(
    name="chat_safe",
    description="no tools",
    tool_policy=ToolPolicy(allowed=NO_TOOLS),
    ctx_budget=4000,
    max_tool_turns=0,
    model=ModelConfig(
        model="ministral-3b-2512",
        temperature=0.2,
        max_tokens=700,
        parallel_tool_calls=False,
    ),
    system_message="Helpful assistant. profile={profile} cwd={cwd}. No tools.",
)

PROFILES["dev_all_tools"] = AgentProfile(
    name="dev_all_tools",
    description="all tools",
    tool_policy=ToolPolicy(allowed=ALL_TOOLS),
    ctx_budget=12000,
    max_tool_turns=12,
    model=ModelConfig(
        model="ministral-3b-2512",
        temperature=0.1,
        max_tokens=1024,
        parallel_tool_calls=False,
    ),
    system_message="Concise coding assistant. profile={profile} cwd={cwd}. Use tools when helpful.",
)


def get_profile(name: str | None) -> AgentProfile:
    return PROFILES.get((name or "").strip(), PROFILES["default"])
