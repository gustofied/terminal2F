import hashlib
import math
import threading
import time
import rerun.blueprint as rrb
import rerun as rr


_initialized = False
_started = False
_blueprint_sent = False

ROOT = "t2f"
TABLE_ROOT = f"{ROOT}/tables"
AGENT_STATE_TABLE = f"{TABLE_ROOT}/agent_state"
AGENT_SPEC_TABLE = f"{TABLE_ROOT}/agent_spec"

USER = [80, 160, 255, 255]
ASSISTANT = [120, 220, 120, 255]
TOOL = [255, 200, 80, 255]
EVENT = [180, 180, 180, 255]

IDLE = [160, 160, 160, 255]
ACTIVE = [120, 220, 120, 255]

_frame = 0
_step = 0

_agents: dict[tuple[str, str, str], dict] = {}
_spec_logged: set[tuple[str, str, str]] = set()


def new_location(center_x, center_y, x, y, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    translated_x = x - center_x
    translated_y = y - center_y
    new_x = translated_x * cos_a - translated_y * sin_a + center_x
    new_y = translated_x * sin_a + translated_y * cos_a + center_y
    return new_x, new_y


def _base(episode_id: str, agent_name: str, instance_id: str) -> str:
    return f"{ROOT}/episodes/{episode_id}/agents/{agent_name}/instances/{instance_id}"


def set_step(step: int) -> None:
    global _step
    _step = step
    rr.set_time("bench_step", sequence=step)


def tick(delta: int = 1) -> int:
    global _step
    new_step = int(_step) + int(delta or 1)
    set_step(new_step)
    return new_step


def _st(episode_id: str, agent_name: str, instance_id: str) -> dict:
    k = (episode_id, agent_name, instance_id)
    st = _agents.get(k)
    if st is None:
        seed = f"{episode_id}:{agent_name}:{instance_id}"
        n = int.from_bytes(
            hashlib.blake2b(seed.encode(), digest_size=8).digest(), "little"
        )
        r = 6.0 + (n % 7) * 1.1
        a0 = (n % 360) * (math.pi / 180.0)
        st = _agents[k] = {
            "n": n,
            "bx": r * math.cos(a0),
            "by": r * math.sin(a0),
            "frac": 0.0,
            "active_until": -1,
            "agent_step": 0,
        }
    return st


def register_agent(episode_id: str, agent_name: str, instance_id: str) -> None:
    _st(episode_id, agent_name, instance_id)


def log_agent_spec(
    episode_id: str,
    agent_name: str,
    instance_id: str,
    *,
    step: int,
    model: str | None = None,
    max_context_length: int | None = None,
    system_message: str | None = None,
    tools_installed: list[str] | None = None,
) -> None:
    k = (episode_id, agent_name, instance_id)
    if k in _spec_logged:
        return

    set_step(step)

    sm = system_message or ""
    sm_preview = sm[:300] + ("…" if len(sm) > 300 else "")
    sm_hash = (
        hashlib.blake2b(sm.encode("utf-8"), digest_size=8).hexdigest() if sm else ""
    )

    rr.log(
        AGENT_SPEC_TABLE,
        rr.AnyValues(
            episode_id=episode_id,
            agent_name=agent_name,
            instance_id=instance_id,
            agent_key=f"{agent_name}:{instance_id}",
            model=model or "",
            max_context_length=int(max_context_length or 0),
            system_message_preview=sm_preview,
            system_message_hash=sm_hash,
            tools_installed=",".join(sorted(tools_installed or [])),
        ),
    )

    _spec_logged.add(k)


def log_agent_state(
    episode_id: str,
    agent_name: str,
    instance_id: str,
    *,
    step: int,
    agent_step: int,
    env_name: str = "",
    model: str | None = None,
    prompt_tokens_last: int = 0,
    prompt_tokens_max: int = 0,
    context_budget: int | None = None,
    context_limit: int | None = None,
    warned_budget: bool = False,
    tools_allowed: list[str] | None = None,
    tools_exposed: list[str] | None = None,
    tool_calls: int = 0,
    tool_errors: int = 0,
    tool_turns: int = 0,
    llm_calls: int = 0,
    user_chars: int = 0,
    assistant_chars: int = 0,
    last_tool_name: str = "",
    last_tool_error: str = "",
) -> None:
    set_step(step)

    cl = int(context_limit or 0)
    frac = float(prompt_tokens_max / cl) if cl > 0 else 0.0

    rr.log(
        AGENT_STATE_TABLE,
        rr.AnyValues(
            episode_id=episode_id,
            bench_step=int(step),
            env_name=str(env_name or ""),
            agent_name=agent_name,
            instance_id=instance_id,
            agent_key=f"{agent_name}:{instance_id}",
            agent_step=int(agent_step),
            model=model or "",
            prompt_tokens_last=int(prompt_tokens_last),
            prompt_tokens_max=int(prompt_tokens_max),
            context_budget=int(context_budget) if context_budget is not None else -1,
            context_limit=int(context_limit or 0),
            context_frac=float(frac),
            warned_budget=bool(warned_budget),
            tools_allowed=",".join(sorted(tools_allowed or [])),
            tools_exposed=",".join(sorted(tools_exposed or [])),
            tool_calls=int(tool_calls),
            tool_errors=int(tool_errors),
            tool_turns=int(tool_turns),
            llm_calls=int(llm_calls),
            user_chars=int(user_chars),
            assistant_chars=int(assistant_chars),
            last_tool_name=str(last_tool_name or ""),
            last_tool_error=str((last_tool_error or "")[:300]),
        ),
    )


def _draw(frame: int):
    cx = cy = 0.0
    angle = frame * 0.05

    pts, cols, rads, labels = [], [], [], []
    for (_episode_id, name, iid), st in list(_agents.items()):
        x, y = new_location(cx, cy, st["bx"], st["by"], angle)

        base = 0.04 + 0.25 * st["frac"]
        pulse = 0.025 * math.sin(frame * (0.15 + (st["n"] % 10) * 0.03))
        rad = max(base + pulse, 1.01)

        pts.append([x, y])
        cols.append(ACTIVE if frame <= st["active_until"] else IDLE)
        rads.append(rad)
        labels.append(f"{name}:{iid} | bench={_step} | agent={st.get('agent_step', 0)}")

    rr.log(
        f"{ROOT}/swarm/points",
        rr.Points2D(pts, colors=cols, radii=rads, labels=labels, show_labels=True),
    )


def _anim():
    global _frame
    while True:
        rr.set_time("frame", sequence=_frame)
        rr.set_time("bench_step", sequence=_step)
        if _agents:
            _draw(_frame)
        _frame += 1
        time.sleep(1 / 30)


def _send_default_blueprint() -> None:
    global _blueprint_sent
    if _blueprint_sent:
        return
    if rrb is None:
        return

    bp = rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(
                origin=f"{ROOT}/swarm",
                name="Swarm",
                contents=f"{ROOT}/swarm/**",
            ),
            rrb.TextLogView(
                origin=f"{ROOT}/episodes",
                name="Episode logs",
                contents=f"{ROOT}/episodes/**",
            ),
            rrb.TimeSeriesView(
                origin=f"{ROOT}/episodes",
                name="Usage",
                contents=f"{ROOT}/episodes/**/usage/**",
            ),
            rrb.DataframeView(
                origin=f"{ROOT}/tables",
                name="Tables (DF)",
                query=rrb.archetypes.DataframeQuery(
                    timeline="bench_step",
                    apply_latest_at=True,
                ),
            ),
            grid_columns=2,
            name="t2f dashboard",
        ),
        collapse_panels=True,
    )

    rr.send_blueprint(bp)
    _blueprint_sent = True


def init(
    app_id: str = "the_agent_logs",
    *,
    spawn: bool = True,
    send_blueprint: bool = True,
    log_config_path: str | None = None,
) -> None:
    global _initialized, _started
    if _initialized:
        return

    rr.init(app_id, spawn=spawn)

    rr.log(
        f"{ROOT}/swarm/origin",
        rr.Boxes2D(
            mins=[-1, -1],
            sizes=[2, 2],
            labels=["terminal2F"],
            show_labels=True,
        ),
        static=True,
    )

    if send_blueprint:
        _send_default_blueprint()

    if not _started:
        _started = True
        threading.Thread(target=_anim, daemon=True).start()

    if log_config_path:
        from .logging.mylogger import setup_logging
        setup_logging(log_config_path)

    _initialized = True


def on_event(episode_id: str, agent_name: str, instance_id: str, step: int, text: str) -> None:
    set_step(step)
    rr.log(
        f"{_base(episode_id, agent_name, instance_id)}/events",
        rr.TextLog(text, level=rr.TextLogLevel.INFO, color=EVENT),
    )

    if "cleared" in (text or "").lower():
        st = _st(episode_id, agent_name, instance_id)
        st["frac"] = 0.0
        st["active_until"] = -1
        st["agent_step"] = 0


def on_turn(
    episode_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    *,
    agent_step: int,
    user_message: str,
) -> None:
    set_step(step)
    rr.log(
        f"{_base(episode_id, agent_name, instance_id)}/conversation",
        rr.TextLog(f"user: {user_message}", level=rr.TextLogLevel.INFO, color=USER),
    )
    st = _st(episode_id, agent_name, instance_id)
    st["agent_step"] = agent_step
    st["active_until"] = _frame + 20


def on_tool_call(
    episode_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    function_name: str,
    function_params: dict,
) -> None:
    set_step(step)
    rr.log(
        f"{_base(episode_id, agent_name, instance_id)}/tool_calls",
        rr.TextLog(
            f"{function_name}({function_params})", level=rr.TextLogLevel.INFO, color=TOOL
        ),
    )


def on_tool_result(
    episode_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    function_name: str,
    function_result: str,
) -> None:
    set_step(step)
    rr.log(
        f"{_base(episode_id, agent_name, instance_id)}/tool_results",
        rr.TextLog(f"{function_name} -> {function_result}", level=rr.TextLogLevel.INFO, color=TOOL),
    )


def on_assistant(
    episode_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    content: str,
) -> None:
    set_step(step)
    rr.log(
        f"{_base(episode_id, agent_name, instance_id)}/conversation",
        rr.TextLog(f"assistant: {content}", level=rr.TextLogLevel.INFO, color=ASSISTANT),
    )


def on_usage(
    episode_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    prompt_tokens: int,
    *,
    context_limit: int,
) -> None:
    set_step(step)
    rr.log(
        f"{_base(episode_id, agent_name, instance_id)}/usage/context_length",
        rr.Scalars(prompt_tokens),
    )
    st = _st(episode_id, agent_name, instance_id)
    st["frac"] = min(prompt_tokens / max(context_limit, 1), 1.0)
