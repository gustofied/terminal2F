import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .. import control_tower
from ..control_tower import RunContext
from ..agent_profiles import AgentProfile, compile_tools, get_profile, tool_name
from ..tools import names_to_functions

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


def run_agent(
    agent,
    user_message: str,
    *,
    memory: RunnerMemory,
    run: RunContext,
    ui=None,
    tool_schemas: list[dict] | None = None,
    context_budget: int | None = None,
    max_turns: int | None = None,
):
    profile: AgentProfile = getattr(agent, "profile", None) or get_profile("default")
    context_budget = profile.ctx_budget if context_budget is None else context_budget
    max_turns = profile.max_tool_turns if max_turns is None else int(max_turns)

    recording_id = run.recording_id
    run_id = run.run_id
    step = run.tick()

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

    control_tower.register_agent(recording_id, run_id, name, iid)

    installed_names = sorted(_schema_names(tools_installed))
    allowed_names = sorted(tool_names_allowed)
    exposed_names = sorted(_schema_names(tools_exposed))

    control_tower.log_agent_spec(
        recording_id,
        run_id,
        name,
        iid,
        step=step,
        model=getattr(agent, "model", None),
        max_context_length=getattr(agent, "max_context_length", None),
        system_message=getattr(agent, "system_message", None),
        tools_installed=installed_names,
    )

    msgs.append({"role": "user", "content": user_message})
    control_tower.on_turn(
        recording_id,
        run_id,
        name,
        iid,
        step,
        agent_step=agent_step,
        user_message=user_message,
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
        control_tower.on_event(recording_id, run_id, name, iid, step, txt)
        _ui_call("on_event", txt)

    context_window = 0
    warned = False
    llm_calls = 0
    last_prompt_tokens = 0

    tool_calls = 0
    tool_errors = 0
    tool_turns = 0
    last_tool_name = ""
    last_tool_error = ""

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

    turns = 0
    while assistant.get("tool_calls"):
        turns += 1
        tool_turns = turns
        if turns > max_turns:
            raise RuntimeError("Max turns reached without a final answer.")

        for tool_call in assistant["tool_calls"]:
            tool_calls += 1

            fn_block = tool_call.get("function") or {}
            function_name = fn_block.get("name") or ""
            raw_args = fn_block.get("arguments") or "{}"
            last_tool_name = function_name

            if function_name not in tool_names_allowed:
                function_params = {}
                function_result = f"error: tool '{function_name}' not allowed by profile.tool_policy"
                tool_errors += 1
                last_tool_error = function_result
            else:
                try:
                    function_params = json.loads(raw_args)
                    if not isinstance(function_params, dict):
                        raise ValueError("tool arguments must decode to an object")
                except Exception as err:
                    function_params = {}
                    function_result = f"error: {err}"
                    tool_errors += 1
                    last_tool_error = function_result
                else:
                    control_tower.on_tool_call(
                        recording_id, run_id, name, iid, step, function_name, function_params
                    )
                    _ui_call("on_tool_call", function_name, function_params)
                    function_result = _run_tool(function_name, function_params)

                    if (function_result or "").startswith("error:"):
                        tool_errors += 1
                        last_tool_error = function_result

            _ui_call("on_tool_result", function_name, function_result)
            control_tower.on_tool_result(
                recording_id, run_id, name, iid, step, function_name, function_result
            )

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

    final_text = assistant.get("content", "") or ""
    control_tower.on_assistant(recording_id, run_id, name, iid, step, final_text)

    control_tower.on_usage(
        recording_id,
        run_id,
        name,
        iid,
        step,
        context_window,
        context_limit=agent.max_context_length,
    )

    control_tower.log_agent_state(
        recording_id,
        run_id,
        name,
        iid,
        step=step,
        agent_step=agent_step,
        profile_name=profile.name,
        model=getattr(agent, "model", None),
        prompt_tokens_last=int(last_prompt_tokens or 0),
        prompt_tokens_max=int(context_window or 0),
        context_budget=context_budget,
        context_limit=getattr(agent, "max_context_length", None),
        warned_budget=bool(warned),
        tools_allowed=allowed_names,
        tools_exposed=exposed_names,
        tool_calls=int(tool_calls),
        tool_errors=int(tool_errors),
        tool_turns=int(tool_turns),
        llm_calls=int(llm_calls),
        user_chars=len(user_message or ""),
        assistant_chars=len(final_text or ""),
        last_tool_name=str(last_tool_name or ""),
        last_tool_error=str(last_tool_error or ""),
    )

    return response
