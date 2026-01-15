from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ToolSchema = dict[str, Any]


@dataclass(frozen=True)
class Rules:
    ctx_budget: int | None = 5000
    max_tool_turns: int = 10
    tool_policy: set[str] = field(default_factory=set)

    def allows(self, tool_name: str) -> bool:
        return bool(tool_name) and ("*" in self.tool_policy or tool_name in self.tool_policy)


@dataclass(frozen=True)
class AgentPolicies:
    model: str = "ministral-3b-2512"
    temperature: float = 0.1
    max_tokens: int = 1024
    parallel_tool_calls: bool = False

    system_message: str = "Concise coding assistant. env={env} cwd={cwd}"


@dataclass(frozen=True)
class Env:
    name: str
    rules: Rules = field(default_factory=Rules)
    agent: AgentPolicies = field(default_factory=AgentPolicies)

    def render_system_message(self, *, cwd: str) -> str:
        return self.agent.system_message.format(env=self.name, cwd=cwd)


def tool_name(schema: ToolSchema) -> str:
    return schema["function"]["name"]


def compile_tools(
    *,
    env: Env,
    installed_tools: list[ToolSchema],
    requested_tools: list[ToolSchema] | None,
) -> tuple[set[str], list[ToolSchema]]:
    """
    Single truth:
      tools_exposed_to_model == tools_allowed_to_execute
    """
    installed = installed_tools or []
    if not env.rules.tool_policy:
        return set(), []

    basis = installed if requested_tools is None else (requested_tools or [])

    installed_names = {tool_name(t) for t in installed}
    exposed = [t for t in basis if tool_name(t) in installed_names and env.rules.allows(tool_name(t))]

    allowed = {tool_name(t) for t in exposed}
    return allowed, exposed

ENVS: dict[str, Env] = {}

ENVS["default"] = Env(
    name="default",
    rules=Rules(
        ctx_budget=5000,
        max_tool_turns=10,
        tool_policy={"retrieve_payment_status", "retrieve_payment_date"},
    ),
    agent=AgentPolicies(
        model="ministral-3b-2512",
        temperature=0.1,
        max_tokens=1024,
        parallel_tool_calls=False,
        system_message="Concise coding assistant. env={env} cwd={cwd}",
    ),
)

ENVS["chat_safe"] = Env(
    name="chat_safe",
    rules=Rules(
        ctx_budget=4000,
        max_tool_turns=0,
        tool_policy=set(),  
    ),
    agent=AgentPolicies(
        temperature=0.2,
        max_tokens=700,
        parallel_tool_calls=False,
        system_message="Helpful assistant. env={env} cwd={cwd}. No tools.",
    ),
)

ENVS["dev_all_tools"] = Env(
    name="dev_all_tools",
    rules=Rules(
        ctx_budget=12000,
        max_tool_turns=12,
        tool_policy={"*"},  
    ),
    agent=AgentPolicies(
        temperature=0.1,
        max_tokens=1024,
        parallel_tool_calls=False,
        system_message="Concise coding assistant. env={env} cwd={cwd}. Use tools when helpful.",
    ),
)


def get_env(name: str | None) -> Env:
    return ENVS.get((name or "").strip(), ENVS["default"])
