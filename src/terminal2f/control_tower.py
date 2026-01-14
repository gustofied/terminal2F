# control_tower.py
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

USER = [80, 160, 255, 255]
ASSISTANT = [120, 220, 120, 255]
TOOL = [255, 200, 80, 255]
EVENT = [180, 180, 180, 255]

IDLE = [160, 160, 160, 255]
ACTIVE = [120, 220, 120, 255]

_frame = 0
_step = 0

# key: (episode_id, agent_name, instance_id)
_agents: dict[tuple[str, str, str], dict] = {}


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


def _st(episode_id: str, agent_name: str, instance_id: str) -> dict:
    k = (episode_id, agent_name, instance_id)
    st = _agents.get(k)
    if st is None:
        seed = f"{episode_id}:{agent_name}:{instance_id}"
        n = int.from_bytes(hashlib.blake2b(seed.encode(), digest_size=8).digest(), "little")
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


def _draw(frame: int):
    cx = cy = 0.0
    angle = frame * 0.05

    pts, cols, rads, labels = [], [], [], []
    for (episode_id, name, iid), st in _agents.items():
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
    """Send a simple 2x2 dashboard blueprint.

    Safe to call multiple times; it will only send once per process.
    """
    global _blueprint_sent
    if _blueprint_sent:
        return
    if rrb is None:
        return

    # A dashboard that works for any episode_id by targeting your stable roots.
    bp = rrb.Blueprint(
        rrb.Grid(
            # 1) Swarm visualization
            rrb.Spatial2DView(
                origin=f"{ROOT}/swarm",
                name="Swarm",
                contents=f"{ROOT}/swarm/**",
            ),
            # 2) Text logs (conversation / tool calls / events)
            rrb.TextLogView(
                origin=f"{ROOT}/episodes",
                name="Episode logs",
                contents=f"{ROOT}/episodes/**",
            ),
            # 3) Usage scalars over time
            rrb.TimeSeriesView(
                origin=f"{ROOT}/episodes",
                name="Usage",
                contents=f"{ROOT}/episodes/**/usage/**",
            ),
            # 4) Dataframe view of everything (your “grid container”)
            rrb.DataframeView(
                origin="/",
                name="All data (DF)",
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


def init(app_id: str = "the_agent_logs", *, spawn: bool = True, send_blueprint: bool = True) -> None:
    global _initialized, _started
    if _initialized:
        return

    rr.init(app_id, spawn=spawn)

    # Establish the swarm space (static).
    rr.log(
        f"{ROOT}/swarm/origin",
        rr.Boxes2D(mins=[-1, -1], sizes=[2, 2], labels=["terminal2F"], show_labels=True),
        static=True,
    )

    # Send the default dashboard layout.
    if send_blueprint:
        _send_default_blueprint()

    # Start background animation thread once.
    if not _started:
        _started = True
        threading.Thread(target=_anim, daemon=True).start()

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
        rr.TextLog(f"{function_name}({function_params})", level=rr.TextLogLevel.INFO, color=TOOL),
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
