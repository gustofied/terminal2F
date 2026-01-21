import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import rerun as rr
import rerun.blueprint as rrb

_initialized = False
_started = False
_blueprint_sent = False

ROOT = "t2f"
RUNS_ROOT = f"{ROOT}/runs"

TABLE_ROOT = f"{ROOT}/tables"
RUN_SPEC_TABLE = f"{TABLE_ROOT}/run_spec"
AGENT_STATE_TABLE = f"{TABLE_ROOT}/agent_state"
AGENT_SPEC_TABLE = f"{TABLE_ROOT}/agent_spec"

USER = [80, 160, 255, 255]
ASSISTANT = [120, 220, 120, 255]
TOOL = [255, 200, 80, 255]
EVENT = [180, 180, 180, 255]

IDLE = [160, 160, 160, 255]
ACTIVE = [120, 220, 120, 255]

_frame = 0

_agents: dict[tuple[str, str, str], dict] = {}
_spec_logged: set[tuple[str, str, str]] = set()
_run_spec_logged: set[tuple[str, str]] = set()  # (recording_id, run_id)

_bench_step = 0
_bench_lock = threading.Lock()


def _new_id(n: int = 8) -> str:
    return uuid.uuid4().hex[: int(n)]


def _bench_now() -> int:
    with _bench_lock:
        return int(_bench_step)


def _tick_bench(delta: int = 1) -> int:
    global _bench_step
    d = int(delta or 1)
    if d <= 0:
        d = 1
    with _bench_lock:
        _bench_step += d
        rr.set_time("bench_step", sequence=int(_bench_step))
        return int(_bench_step)


@dataclass
class RunContext:
    recording_id: str
    run_id: str
    run_step: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def tick(self, delta: int = 1) -> int:
        d = int(delta or 1)
        if d <= 0:
            d = 1
        with self._lock:
            self.run_step += d
        return _tick_bench(d)


def new_location(center_x, center_y, x, y, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    translated_x = x - center_x
    translated_y = y - center_y
    new_x = translated_x * cos_a - translated_y * sin_a + center_x
    new_y = translated_x * sin_a + translated_y * cos_a + center_y
    return new_x, new_y


def _base(run_id: str, agent_name: str, instance_id: str) -> str:
    return f"{RUNS_ROOT}/{run_id}/agents/{agent_name}/instances/{instance_id}"


def _st(run_id: str, agent_name: str, instance_id: str) -> dict:
    k = (run_id, agent_name, instance_id)
    st = _agents.get(k)
    if st is None:
        seed = f"{run_id}:{agent_name}:{instance_id}"
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
            "bench_step": 0,
        }
    return st


def register_agent(recording_id: str, run_id: str, agent_name: str, instance_id: str) -> None:
    _ = recording_id
    _st(run_id, agent_name, instance_id)


def log_run_spec(
    recording_id: str,
    run_id: str,
    *,
    name: str,
    runner_name: str,
    profile_name: str,
    agent_models: dict[str, str],
    agent_tools: dict[str, list[str]],
) -> None:
    """
    Logs a single row per run so you can see config in Rerun tables.
    """
    k = (recording_id, run_id)
    if k in _run_spec_logged:
        return

    rr.set_time("bench_step", sequence=_bench_now())

    rr.log(
        RUN_SPEC_TABLE,
        rr.AnyValues(
            recording_id=recording_id,
            run_id=run_id,
            run_name=str(name or ""),
            runner_name=str(runner_name or ""),
            profile_name=str(profile_name or ""),
            agents=",".join(sorted(agent_models.keys())),
            agent_models=json.dumps(agent_models, ensure_ascii=False),
            agent_tools=json.dumps(agent_tools, ensure_ascii=False),
            tools_total=int(sum(len(v) for v in agent_tools.values())),
        ),
    )

    _run_spec_logged.add(k)


def log_agent_spec(
    recording_id: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    *,
    step: int,
    model: str | None = None,
    max_context_length: int | None = None,
    system_message: str | None = None,
    tools_installed: list[str] | None = None,
) -> None:
    k = (run_id, agent_name, instance_id)
    if k in _spec_logged:
        return

    rr.set_time("bench_step", sequence=int(step))

    sm = system_message or ""
    sm_preview = sm[:300] + ("…" if len(sm) > 300 else "")
    sm_hash = (
        hashlib.blake2b(sm.encode("utf-8"), digest_size=8).hexdigest() if sm else ""
    )

    rr.log(
        AGENT_SPEC_TABLE,
        rr.AnyValues(
            recording_id=recording_id,
            run_id=run_id,
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
    recording_id: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    *,
    step: int,
    agent_step: int,
    profile_name: str = "",
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
    llm_calls: int = 0,
    user_chars: int = 0,
    assistant_chars: int = 0,
    last_tool_name: str = "",
    last_tool_error: str = "",
) -> None:
    rr.set_time("bench_step", sequence=int(step))

    cl = int(context_limit or 0)
    frac = float(prompt_tokens_max / cl) if cl > 0 else 0.0

    rr.log(
        AGENT_STATE_TABLE,
        rr.AnyValues(
            recording_id=recording_id,
            run_id=run_id,
            bench_step=int(step),
            profile_name=str(profile_name or ""),
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
    for (run_id, name, iid), st in list(_agents.items()):
        x, y = new_location(cx, cy, st["bx"], st["by"], angle)

        base = 0.04 + 0.25 * st["frac"]
        pulse = 0.025 * math.sin(frame * (0.15 + (st["n"] % 10) * 0.03))
        rad = max(base + pulse, 1.01)

        pts.append([x, y])
        cols.append(ACTIVE if frame <= st["active_until"] else IDLE)
        rads.append(rad)

        bench_step = int(st.get("bench_step", 0))
        agent_step = int(st.get("agent_step", 0))
        labels.append(f"{run_id} | {name}:{iid} | bench={bench_step} | agent={agent_step}")

    rr.log(
        f"{ROOT}/swarm/points",
        rr.Points2D(pts, colors=cols, radii=rads, labels=labels, show_labels=True),
    )


def _anim():
    global _frame
    while True:
        rr.set_time("frame", sequence=_frame)
        if _agents:
            _draw(_frame)
        _frame += 1
        time.sleep(1 / 30)


def _send_default_blueprint() -> None:
    global _blueprint_sent
    if _blueprint_sent or rrb is None:
        return

    bp = rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial2DView(
                origin=f"{ROOT}/swarm",
                name="Swarm",
                contents=f"{ROOT}/swarm/**",
            ),
            rrb.TextLogView(
                origin=f"{RUNS_ROOT}",
                name="Run logs",
                contents=f"{RUNS_ROOT}/**",
            ),
            rrb.TimeSeriesView(
                origin=RUNS_ROOT,
                name="Usage",
                contents=[f"+ {RUNS_ROOT}/**/usage/context_length"],
            ),
            rrb.DataframeView(
                origin=f"{TABLE_ROOT}",
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
    recording_id: str,
    spawn: bool = True,
    send_blueprint: bool = True,
    log_config_path: str | None = None,
) -> None:
    global _initialized, _started

    if _initialized:
        return

    rr.init(app_id, recording_id=recording_id, spawn=spawn)

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

    if log_config_path is None:
        default_cfg = Path(__file__).resolve().parent / "mylogger" / "config.json"
        if default_cfg.is_file():
            log_config_path = str(default_cfg)

    if log_config_path:
        from .mylogger.mylogger import setup_logging
        setup_logging(log_config_path)

    _initialized = True


def start_run(
    *,
    app_id: str = "the_agent_logs",
    spawn: bool = True,
    send_blueprint: bool = True,
    recording_id: str | None = None,
    run_id: str | None = None,
) -> RunContext:
    rid = (recording_id or _new_id()).strip()
    init(app_id=app_id, recording_id=rid, spawn=spawn, send_blueprint=send_blueprint)

    run_id = (run_id or _new_id()).strip()
    return RunContext(recording_id=rid, run_id=run_id, run_step=0)


def start_new_run(parent: RunContext, *, run_id: str | None = None) -> RunContext:
    return RunContext(
        recording_id=parent.recording_id,
        run_id=(run_id or _new_id()).strip(),
        run_step=0,
    )


@dataclass
class Recording:
    recording_id: str
    app_id: str = "the_agent_logs"
    spawn: bool = True
    send_blueprint: bool = True

    def add_run(
        self,
        *,
        name: str,
        profile,
        runner_name: str = "loop",
        agents: dict[str, dict[str, Any]] | None = None,
        task: Callable[["Run"], None] | None = None,
        run_id: str | None = None,
        ui: Any | None = None,
        tool_schemas: list[dict] | None = None,
    ) -> "Run":
        agents = agents or {}

        ctx = RunContext(
            recording_id=self.recording_id,
            run_id=(run_id or _new_id()).strip(),
            run_step=0,
        )

        run = Run(
            name=name,
            context=ctx,
            runner_name=runner_name,
            profile=profile,
            agents_spec=agents,
            task=task,
            ui=ui,
            tool_schemas=tool_schemas,
        )

        return run

    def play(self, runs: list["Run"], *, n: int = 1, interval_s: float = 0.0) -> None:
        runs = list(runs or [])
        if not runs:
            return

        n = max(int(n or 0), 0)

        for epoch in range(n):
            rr.set_time("epoch", sequence=int(epoch))

            for r in runs:
                r.step()

            if interval_s:
                time.sleep(float(interval_s))


@dataclass
class Run:
    name: str
    context: RunContext
    runner_name: str
    profile: Any
    agents_spec: dict[str, dict[str, Any]]
    task: Callable[["Run"], None] | None = None
    ui: Any | None = None
    tool_schemas: list[dict] | None = None

    def __post_init__(self):
        from .agent import Agent
        from .runners import get_runner
        from .agent_profiles import tool_name

        self.runner = get_runner(self.runner_name)

        self.agents: dict[str, Any] = {}
        self.memories: dict[str, Any] = {}

        agent_models: dict[str, str] = {}
        agent_tools: dict[str, list[str]] = {}

        for agent_name, spec in (self.agents_spec or {}).items():
            tools_installed = spec.get("tools") or []
            instance_id = spec.get("instance_id") or agent_name
            model = spec.get("model")

            agent = Agent(
                tools_installed=tools_installed,
                profile=self.profile,
                model=model,
                name=agent_name,
                instance_id=instance_id,
            )
            mem = self.runner.new_memory(agent)

            self.agents[agent_name] = agent
            self.memories[agent_name] = mem

            agent_models[agent_name] = str(getattr(agent, "model", "") or "")
            agent_tools[agent_name] = sorted([tool_name(t) for t in (tools_installed or [])])

        log_run_spec(
            self.context.recording_id,
            self.context.run_id,
            name=self.name,
            runner_name=self.runner_name,
            profile_name=getattr(self.profile, "name", "") or "",
            agent_models=agent_models,
            agent_tools=agent_tools,
        )

    def turn(self, agent_name: str, user_message: str):
        if agent_name not in self.agents:
            raise KeyError(f"Unknown agent '{agent_name}'. Known={list(self.agents)}")

        agent = self.agents[agent_name]
        mem = self.memories[agent_name]

        return self.runner(
            agent,
            user_message,
            memory=mem,
            run=self.context,
            ui=self.ui,
            tool_schemas=self.tool_schemas,
        )

    def step(self) -> None:
        if not callable(self.task):
            raise RuntimeError(f"Run '{self.name}' has no task set")
        self.task(self)


def start_recording(
    *,
    recording_id: str,
    app_id: str = "the_agent_logs",
    spawn: bool = True,
    send_blueprint: bool = True,
) -> Recording:
    init(
        app_id=app_id,
        recording_id=recording_id,
        spawn=spawn,
        send_blueprint=send_blueprint,
    )

    return Recording(
        recording_id=recording_id,
        app_id=app_id,
        spawn=spawn,
        send_blueprint=send_blueprint,
    )


def on_event(
    recording_id: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    text: str,
) -> None:
    rr.set_time("bench_step", sequence=int(step))
    rr.log(
        f"{_base(run_id, agent_name, instance_id)}/events",
        rr.TextLog(text, level=rr.TextLogLevel.INFO, color=EVENT),
    )

    if "cleared" in (text or "").lower():
        st = _st(run_id, agent_name, instance_id)
        st["frac"] = 0.0
        st["active_until"] = -1
        st["agent_step"] = 0
        st["bench_step"] = int(step)


def on_turn(
    recording_id: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    *,
    agent_step: int,
    user_message: str,
) -> None:
    rr.set_time("bench_step", sequence=int(step))
    rr.log(
        f"{_base(run_id, agent_name, instance_id)}/conversation",
        rr.TextLog(f"user: {user_message}", level=rr.TextLogLevel.INFO, color=USER),
    )
    st = _st(run_id, agent_name, instance_id)
    st["agent_step"] = int(agent_step)
    st["bench_step"] = int(step)
    st["active_until"] = _frame + 20


def on_tool_call(
    recording_id: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    function_name: str,
    function_params: dict,
) -> None:
    rr.set_time("bench_step", sequence=int(step))
    rr.log(
        f"{_base(run_id, agent_name, instance_id)}/tool_calls",
        rr.TextLog(
            f"{function_name}({function_params})", level=rr.TextLogLevel.INFO, color=TOOL
        ),
    )
    st = _st(run_id, agent_name, instance_id)
    st["bench_step"] = int(step)


def on_tool_result(
    recording_id: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    function_name: str,
    function_result: str,
) -> None:
    rr.set_time("bench_step", sequence=int(step))
    rr.log(
        f"{_base(run_id, agent_name, instance_id)}/tool_results",
        rr.TextLog(f"{function_name} -> {function_result}", level=rr.TextLogLevel.INFO, color=TOOL),
    )
    st = _st(run_id, agent_name, instance_id)
    st["bench_step"] = int(step)


def on_assistant(
    recording_id: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    content: str,
) -> None:
    rr.set_time("bench_step", sequence=int(step))
    rr.log(
        f"{_base(run_id, agent_name, instance_id)}/conversation",
        rr.TextLog(f"assistant: {content}", level=rr.TextLogLevel.INFO, color=ASSISTANT),
    )
    st = _st(run_id, agent_name, instance_id)
    st["bench_step"] = int(step)


def on_usage(
    recording_id: str,
    run_id: str,
    agent_name: str,
    instance_id: str,
    step: int,
    prompt_tokens: int,
    *,
    context_limit: int,
) -> None:
    rr.set_time("bench_step", sequence=int(step))
    rr.log(
        f"{_base(run_id, agent_name, instance_id)}/usage/context_length",
        rr.Scalars(prompt_tokens),
    )
    st = _st(run_id, agent_name, instance_id)
    st["frac"] = min(prompt_tokens / max(context_limit, 1), 1.0)
    st["bench_step"] = int(step)
