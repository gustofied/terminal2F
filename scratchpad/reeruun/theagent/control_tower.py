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


def on_turn(agent_key: str, turn_idx: int, user_message: str) -> None:
    _set_time(turn_idx)
    rr.log(
        f"agents/{agent_key}/conversation",
        rr.TextLog(f"user: {user_message}", level=rr.TextLogLevel.INFO, color=USER),
    )


def on_tool_call(agent_key: str, turn_idx: int, function_name: str, function_params: dict) -> None:
    _set_time(turn_idx)
    rr.log(
        f"agents/{agent_key}/tool_calls",
        rr.TextLog(
            f"{function_name}({function_params})",
            level=rr.TextLogLevel.INFO,
            color=TOOL,
        ),
    )


def on_assistant(agent_key: str, turn_idx: int, content: str) -> None:
    _set_time(turn_idx)
    rr.log(
        f"agents/{agent_key}/conversation",
        rr.TextLog(f"assistant: {content}", level=rr.TextLogLevel.INFO, color=ASSISTANT),
    )


def on_usage(agent_key: str, turn_idx: int, prompt_tokens: int, *, context_limit: int = 262_144) -> None:
    _set_time(turn_idx)

    rr.log(f"agents/{agent_key}/usage/prompt_tokens", rr.Scalars(prompt_tokens))

    fraction = min(prompt_tokens / context_limit, 1.0)
    base_radius = 10.2
    max_extra = 20.8
    radius = base_radius + max_extra * fraction

    if fraction < 0.5:
        color = [0, 255, 0]
    elif fraction < 0.8:
        color = [255, 200, 0]
    else:
        color = [255, 0, 0]

    rr.log(
        f"agents/{agent_key}/context/circle",
        rr.Points2D([[0.0, 2.0]], radii=[radius], colors=[color]),
    )
