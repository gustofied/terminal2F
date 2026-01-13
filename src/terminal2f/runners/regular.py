import json
import logging
import weakref
from typing import Any

from .. import control_tower
from ..tools import names_to_functions

log = logging.getLogger("app.runner")

# agent -> state (no agent.__dict__ mutation)
_STATE: "weakref.WeakKeyDictionary[object, dict]" = weakref.WeakKeyDictionary()


def reset(agent) -> None:
    """Clear this agent's runner memory."""
    st = _STATE.get(agent)
    if st is not None:
        st["agent_step"] = 0
        st["messages"] = [{"role": "system", "content": agent.system_message}]


def _get_state(agent) -> dict:
    st = _STATE.get(agent)
    if st is None:
        st = {
            "instance_id": getattr(agent, "instance_id", hex(id(agent))[2:]),
            "agent_name": getattr(agent, "name", "agent"),
            "agent_step": 0,
            "messages": [{"role": "system", "content": agent.system_message}],
        }
        _STATE[agent] = st
    return st


def _run_tool(function_name: str, function_params: dict) -> str:
    try:
        fn = names_to_functions[function_name]
    except Exception as err:
        return f"error: {err}"
    try:
        return fn(**function_params)
    except Exception as err:
        return f"error: {err}"


def _msg_to_dict(msg: Any) -> dict:
    # keep messages consistently dict-shaped
    if isinstance(msg, dict):
        out = dict(msg)
    elif hasattr(msg, "model_dump"):
        out = msg.model_dump(exclude_none=True)
    elif hasattr(msg, "dict"):
        out = msg.dict(exclude_none=True)
    else:
        out = {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", "") or "",
        }

    tcs = out.get("tool_calls")
    if tcs is not None:
        dumped = []
        for tc in tcs:
            if isinstance(tc, dict):
                dumped.append(tc)
            elif hasattr(tc, "model_dump"):
                dumped.append(tc.model_dump(exclude_none=True))
            elif hasattr(tc, "dict"):
                dumped.append(tc.dict(exclude_none=True))
        out["tool_calls"] = dumped
    return out


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


def run_agent(
    agent,
    user_message: str,
    *,
    episode_id: str,
    step: int,
    max_turns: int = 10,
    ui=None,
):
    """
    Full transition:
      - episode_id + step REQUIRED (global timeline is external)
      - per-agent memory kept in module state, not agent.__dict__
      - agent_step increments per call for that agent
    """
    st = _get_state(agent)

    instance_id = st["instance_id"]
    agent_name = st["agent_name"]
    messages = st["messages"]

    st["agent_step"] += 1
    agent_step = st["agent_step"]

    control_tower.register_agent(episode_id, agent_name, instance_id)

    messages.append({"role": "user", "content": user_message})

    control_tower.on_turn(
        episode_id,
        agent_name,
        instance_id,
        step,
        agent_step=agent_step,
        user_message=user_message,
    )

    context_window = 0

    def _ui_call(method_name: str, *args):
        if not ui:
            return
        fn = getattr(ui, method_name, None)
        if callable(fn):
            fn(*args)

    # ---- LLM call ----
    response = agent.step(messages)
    context_window = max(context_window, _usage_prompt_tokens(response))

    assistant = _msg_to_dict(response.choices[0].message)
    messages.append(assistant)

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

            try:
                function_params = json.loads(raw_args)
                if not isinstance(function_params, dict):
                    raise ValueError("tool arguments must decode to an object")
            except Exception as err:
                function_params = {}
                function_result = f"error: {err}"
            else:
                control_tower.on_tool_call(
                    episode_id, agent_name, instance_id, step, function_name, function_params
                )
                _ui_call("on_tool_call", function_name, function_params)

                function_result = _run_tool(function_name, function_params)

            _ui_call("on_tool_result", function_name, function_result)
            control_tower.on_tool_result(
                episode_id, agent_name, instance_id, step, function_name, function_result
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": function_name,
                    "content": function_result,
                }
            )

        response = agent.step(messages)
        context_window = max(context_window, _usage_prompt_tokens(response))

        assistant = _msg_to_dict(response.choices[0].message)
        messages.append(assistant)

        text = assistant.get("content", "") or ""
        if text:
            _ui_call("on_assistant_text", text)

    final_text = assistant.get("content", "") or ""
    control_tower.on_assistant(episode_id, agent_name, instance_id, step, final_text)

    control_tower.on_usage(
        episode_id,
        agent_name,
        instance_id,
        step,
        context_window,
        context_limit=agent.max_context_length,
    )

    return response
