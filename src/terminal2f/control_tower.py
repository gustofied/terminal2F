import hashlib
import colorsys
import rerun as rr

_initialized = False

USER = [80, 160, 255, 255]
ASSISTANT = [120, 220, 120, 255]
TOOL = [255, 200, 80, 255]


def init(app_id: str = "the_agent_logs", *, spawn: bool = True) -> None:
    global _initialized
    if _initialized:
        return
    rr.init(app_id, spawn=spawn)
    _initialized = True


def _set_time(turn_idx: int) -> None:
    rr.set_time("turn", sequence=turn_idx)


def _base(agent_name: str, instance_id: str) -> str:
    return f"agents/{agent_name}/instances/{instance_id}"


def _circle_xy(
    agent_name: str,
    instance_id: str,
    cols: int = 30,
    rows: int = 30,
    spacing: float = 3.0,
) -> tuple[float, float]:
    s = f"{agent_name}:{instance_id}".encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()
    n = int.from_bytes(h, "little")

    col = n % cols
    row = (n // cols) % rows

    return col * spacing, row * spacing


def _agent_rgb(agent_name: str, instance_id: str) -> list[int]:
    s = f"{agent_name}:{instance_id}".encode("utf-8")
    h = hashlib.blake2b(s, digest_size=8).digest()
    n = int.from_bytes(h, "little")

    hue = (n % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255)]


def _blend(a: list[int], b: list[int], t: float) -> list[int]:
    return [int(a[i] * (1.0 - t) + b[i] * t) for i in range(3)]


def on_turn(agent_name: str, instance_id: str, turn_idx: int, user_message: str) -> None:
    _set_time(turn_idx)
    rr.log(
        f"{_base(agent_name, instance_id)}/conversation",
        rr.TextLog(f"user: {user_message}", level=rr.TextLogLevel.INFO, color=USER),
    )


def on_tool_call(agent_name: str, instance_id: str, turn_idx: int, function_name: str, function_params: dict) -> None:
    _set_time(turn_idx)
    rr.log(
        f"{_base(agent_name, instance_id)}/tool_calls",
        rr.TextLog(f"{function_name}({function_params})", level=rr.TextLogLevel.INFO, color=TOOL),
    )


def on_assistant(agent_name: str, instance_id: str, turn_idx: int, content: str) -> None:
    _set_time(turn_idx)
    rr.log(
        f"{_base(agent_name, instance_id)}/conversation",
        rr.TextLog(f"assistant: {content}", level=rr.TextLogLevel.INFO, color=ASSISTANT),
    )


def on_usage(agent_name: str, instance_id: str, turn_idx: int, prompt_tokens: int, *, context_limit: int = 262_144) -> None:
    _set_time(turn_idx)

    rr.log(f"{_base(agent_name, instance_id)}/usage/context_length", rr.Scalars(prompt_tokens))

    fraction = min(prompt_tokens / context_limit, 1.0)
    base_radius = 10.2
    max_extra = 20.8
    radius = base_radius + max_extra * fraction

    base_color = _agent_rgb(agent_name, instance_id)

    if fraction < 0.5:
        color = base_color
    elif fraction < 0.8:
        color = _blend(base_color, [255, 200, 0], 0.6)
    else:
        color = [255, 0, 0]

    x, y = _circle_xy(agent_name, instance_id)

    rr.log(
        f"{_base(agent_name, instance_id)}/context/circle",
        rr.Points2D([[x, y]], radii=[radius], colors=[color]),
    )
