import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..agent_profiles import AgentProfile, compile_tools, get_profile, tool_name
from ..tools import names_to_functions
from ..telemetry_rerun import SegmentContext

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


def _schema_names(tool_schemas: list[dict]) -> set[str]:
    return {tool_name(t) for t in (tool_schemas or [])}


def _run_tool(function_name: str, function_params: dict) -> str:
    fn = names_to_functions.get(function_name)
    if not callable(fn):
        return f"error: tool '{function_name}' not found"
    try:
        return fn(**(function_params or {}))
    except Exception as err:
        return f"error: {err}"


def _tool_result_is_error(s: str) -> bool:
    s = s or ""
    if s.startswith("error:"):
        return True
    try:
        obj = json.loads(s)
        return isinstance(obj, dict) and "error" in obj
    except Exception:
        return False


def run_agent(
    agent,
    user_message: str,
    *,
    memory: RunnerMemory,
    ctx: Optional[SegmentContext] = None,
    ui=None,
    tool_schemas: list[dict] | None = None,
    context_budget: int | None = None,
    max_tool_calls: int | None = None,
):
    profile: AgentProfile = getattr(agent, "profile", None) or get_profile("default")
    context_budget = profile.ctx_budget if context_budget is None else context_budget
    max_tool_calls = profile.max_tool_calls if max_tool_calls is None else int(max_tool_calls)

    tools_installed = getattr(agent, "tools_installed", None) or []
    tool_names_allowed, tools_exposed = compile_tools(
        profile=profile,
        installed_tools=tools_installed,
        requested_tools=tool_schemas,
    )

    iid = memory.instance_id
    name = memory.agent_name
    msgs = memory.messages

    memory.agent_step += 1
    agent_step = memory.agent_step

    step = ctx.step() if ctx else agent_step
    rec = ctx.recorder if ctx else None

    installed_names = sorted(_schema_names(tools_installed))
    allowed_names = sorted(tool_names_allowed)
    exposed_names = sorted(_schema_names(tools_exposed))

    if rec:
        rec.agent_spec(
            name,
            iid,
            profile=profile.name,
            model=str(getattr(agent, "model", "") or ""),
            tools_installed=installed_names,
            tools_allowed=allowed_names,
            tools_exposed=exposed_names,
        )

    msgs.append({"role": "user", "content": user_message})

    if rec:
        rec.turn(name, iid, user_message, step=step)

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
        if rec:
            rec.event(txt)
        _ui_call("on_event", txt)

    context_window = 0
    warned = False
    llm_calls = 0
    last_prompt_tokens = 0

    tool_calls = 0
    tool_rounds = 0
    tool_errors = 0

    response = agent.step(msgs, tools_exposed=tools_exposed)
    llm_calls += 1

    pt = _usage_prompt_tokens(response)
    last_prompt_tokens = pt
    context_window = max(context_window, pt)
    if (not warned) and context_budget is not None and pt > context_budget:
        warned = True
        _warn_budget(pt)

    assistant = response.choices[0].message.model_dump(exclude_none=True)
    msgs.append(assistant)

    text = assistant.get("content", "") or ""
    if text:
        _ui_call("on_assistant_text", text)
        if rec:
            rec.assistant(name, iid, text, step=step)

    while assistant.get("tool_calls"):
        tool_rounds += 1

        for tool_call in assistant["tool_calls"]:
            tool_calls += 1
            if tool_calls > max_tool_calls:
                msg = f"error: max_tool_calls reached ({max_tool_calls})"
                if rec:
                    rec.event(msg)
                raise RuntimeError(msg)

            fn_block = tool_call.get("function") or {}
            function_name = fn_block.get("name") or ""
            raw_args = fn_block.get("arguments") or {}

            if function_name not in tool_names_allowed:
                function_params = {}
                function_result = f"error: tool '{function_name}' not allowed by profile.tool_policy"
                tool_errors += 1
            else:
                try:
                    if isinstance(raw_args, str):
                        function_params = json.loads(raw_args or "{}")
                    elif isinstance(raw_args, dict):
                        function_params = raw_args
                    else:
                        function_params = {}

                    if not isinstance(function_params, dict):
                        raise ValueError("tool arguments must be an object")
                except Exception as err:
                    function_params = {}
                    function_result = f"error: {err}"
                    tool_errors += 1
                else:
                    if rec:
                        rec.tool_call(name, iid, function_name, function_params, step=step)
                    _ui_call("on_tool_call", function_name, function_params)

                    function_result = _run_tool(function_name, function_params)

                    if _tool_result_is_error(function_result):
                        tool_errors += 1

            _ui_call("on_tool_result", function_name, function_result)

            if rec:
                rec.tool_result(name, iid, function_name, function_result, step=step)

            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": function_name,
                    "content": function_result,
                }
            )

        response = agent.step(msgs, tools_exposed=tools_exposed)
        llm_calls += 1

        pt = _usage_prompt_tokens(response)
        last_prompt_tokens = pt
        context_window = max(context_window, pt)

        if (not warned) and context_budget is not None and pt > context_budget:
            warned = True
            _warn_budget(pt)

        assistant = response.choices[0].message.model_dump(exclude_none=True)
        msgs.append(assistant)

        text = assistant.get("content", "") or ""
        if text:
            _ui_call("on_assistant_text", text)
            if rec:
                rec.assistant(name, iid, text, step=step)

    final_text = assistant.get("content", "") or ""

    if rec:
        rec.usage(name, iid, int(last_prompt_tokens or 0), step=step)

    return response
