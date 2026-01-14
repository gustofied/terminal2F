import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .. import control_tower
from ..tools import names_to_functions, payment_tools

log = logging.getLogger("app.runner")


@dataclass
class RunnerMemory:
    instance_id: str
    agent_name: str
    agent_step: int = 0
    messages: list[dict] = field(default_factory=list)


def new_memory(agent) -> RunnerMemory:
    iid = getattr(agent, "instance_id", hex(id(agent))[2:])
    name = getattr(agent, "name", "agent")
    return RunnerMemory(
        instance_id=iid,
        agent_name=name,
        messages=[{"role": "system", "content": agent.system_message}],
    )


def reset(agent, memory: RunnerMemory) -> None:
    memory.agent_step = 0
    memory.messages = [{"role": "system", "content": agent.system_message}]


def _usage_prompt_tokens(resp: Any) -> int:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0
    v = getattr(usage, "prompt_tokens", None)
    if v is not None:
        return int(v) or 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens") or 0)
    return 0


def _tool_name(t: dict) -> str:
    return ((t or {}).get("function") or {}).get("name") or ""


def _tool_names(tool_schemas: list[dict]) -> set[str]:
    out: set[str] = set()
    for t in tool_schemas or []:
        n = _tool_name(t)
        if n:
            out.add(n)
    return out


def _keep_tools(tool_schemas: list[dict], allowed_names: set[str]) -> list[dict]:
    return [t for t in (tool_schemas or []) if _tool_name(t) in allowed_names]


def _run_tool(function_name: str, function_params: dict) -> str:
    fn = names_to_functions.get(function_name)
    if not callable(fn):
        return f"error: tool '{function_name}' not found"
    try:
        return fn(**(function_params or {}))
    except Exception as err:
        return f"error: {err}"


def run_agent(
    agent,
    user_message: str,
    *,
    episode_id: str,
    step: int,
    memory: RunnerMemory,
    max_turns: int = 10,
    ui=None,
    # Narrowing hint only. Runner policy still clamps.
    tool_schemas: list[dict] | None = None,
    context_budget: int | None = 5000,
):
    """
    Regular runner:
      - Agent has installed tools.
      - Runner policy decides what tools are allowed.
      - Caller may pass tool_schemas to further narrow, never to expand.
    """

    # Runner policy (hard allowlist)
    policy_tools = payment_tools

    # Installed tools (capability)
    installed_tools = getattr(agent, "tools", None) or []

    # Optional per-call narrowing request (never expands)
    requested_tools = installed_tools if tool_schemas is None else (tool_schemas or [])

    allowed_to_execute = _tool_names(installed_tools) & _tool_names(policy_tools)
    exposed_to_model = _keep_tools(requested_tools, allowed_to_execute)

    iid = memory.instance_id
    name = memory.agent_name
    msgs = memory.messages

    memory.agent_step += 1
    agent_step = memory.agent_step

    control_tower.register_agent(episode_id, name, iid)

    msgs.append({"role": "user", "content": user_message})
    control_tower.on_turn(
        episode_id, name, iid, step, agent_step=agent_step, user_message=user_message
    )

    def _ui_call(method_name: str, *args):
        if not ui:
            return
        fn = getattr(ui, method_name, None)
        if callable(fn):
            fn(*args)

    def _warn_budget(prompt_tokens: int):
        if context_budget is None or prompt_tokens <= context_budget:
            return
        txt = f"⚠️ context budget exceeded: {prompt_tokens}/{context_budget} prompt tokens"
        control_tower.on_event(episode_id, name, iid, step, txt)
        _ui_call("on_event", txt)

    context_window = 0
    warned = False

    # ---- LLM call ----
    response = agent.step(msgs, tools=exposed_to_model)
    pt = _usage_prompt_tokens(response)
    context_window = max(context_window, pt)
    if (not warned) and context_budget is not None and pt > context_budget:
        warned = True
        _warn_budget(pt)

    assistant = response.choices[0].message.model_dump(exclude_none=True)
    msgs.append(assistant)

    text = assistant.get("content", "") or ""
    if text:
        _ui_call("on_assistant_text", text)

    # ---- tool loop ----
    turns = 0
    while assistant.get("tool_calls"):
        turns += 1
        if turns > max_turns:
            raise RuntimeError("Max turns reached without a final answer.")

        for tool_call in assistant["tool_calls"]:
            fn_block = tool_call.get("function") or {}
            function_name = fn_block.get("name") or ""
            raw_args = fn_block.get("arguments") or "{}"

            if function_name not in allowed_to_execute:
                function_params = {}
                function_result = f"error: tool '{function_name}' not allowed by runner policy"
            else:
                try:
                    function_params = json.loads(raw_args)
                    if not isinstance(function_params, dict):
                        raise ValueError("tool arguments must decode to an object")
                except Exception as err:
                    function_params = {}
                    function_result = f"error: {err}"
                else:
                    control_tower.on_tool_call(
                        episode_id, name, iid, step, function_name, function_params
                    )
                    _ui_call("on_tool_call", function_name, function_params)
                    function_result = _run_tool(function_name, function_params)

            _ui_call("on_tool_result", function_name, function_result)
            control_tower.on_tool_result(
                episode_id, name, iid, step, function_name, function_result
            )

            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": function_name,
                    "content": function_result,
                }
            )

        response = agent.step(msgs, tools=exposed_to_model)
        pt = _usage_prompt_tokens(response)
        context_window = max(context_window, pt)
        if (not warned) and context_budget is not None and pt > context_budget:
            warned = True
            _warn_budget(pt)

        assistant = response.choices[0].message.model_dump(exclude_none=True)
        msgs.append(assistant)

        text = assistant.get("content", "") or ""
        if text:
            _ui_call("on_assistant_text", text)

    final_text = assistant.get("content", "") or ""
    control_tower.on_assistant(episode_id, name, iid, step, final_text)

    control_tower.on_usage(
        episode_id,
        name,
        iid,
        step,
        context_window,
        context_limit=agent.max_context_length,
    )

    return response
