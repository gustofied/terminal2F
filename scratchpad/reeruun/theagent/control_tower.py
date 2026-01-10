import rerun as rr

_initialized = False

def init(app_id: str = "the_agent_logs", *, spawn: bool = True) -> None:
    global _initialized
    if _initialized:
        return
    rr.init(app_id, spawn=spawn)
    _initialized = True

def _set_time(turn_idx: int) -> None:
    rr.set_time("turn", sequence=turn_idx)

def on_turn(turn_idx: int, user_message: str) -> None:
    _set_time(turn_idx)
    rr.log("agent/conversation", rr.TextLog(f"user: {user_message}", level=rr.TextLogLevel.INFO))

def on_tool_call(turn_idx: int, function_name: str, function_params: dict) -> None:
    _set_time(turn_idx)
    rr.log("agent/tool_calls", rr.TextLog(f"{function_name}({function_params})", level=rr.TextLogLevel.INFO))

def on_assistant(turn_idx: int, content: str) -> None:
    _set_time(turn_idx)
    rr.log("agent/conversation", rr.TextLog(f"assistant: {content}", level=rr.TextLogLevel.INFO))

def on_context(turn_idx: int, char_len: int, *, limit: int = 16000) -> None:
    _set_time(turn_idx)
    rr.log("context/char_len", rr.Scalars(char_len))

    fraction = min(char_len / limit, 1.0)
    base_radius = 10.2
    max_extra = 20.8
    radius = base_radius + max_extra * fraction

    if fraction < 0.5:
        color = [0, 255, 0]
    elif fraction < 0.8:
        color = [255, 200, 0]
    else:
        color = [255, 0, 0]

    rr.log("context/circle", rr.Points2D([[0.0, 2.0]], radii=[radius], colors=[color]))
