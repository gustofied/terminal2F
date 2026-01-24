from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import rerun as rr
ROOT = "t2f"


@dataclass
class SegmentRecorder:
    """
    One SegmentRecorder == one recording_id == one .rrd file.
    """

    dataset: str
    segment: str
    recording_id: str
    rrd_path: Path

    application_id: str = "terminal2f"
    spawn_viewer: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rrd_path.parent.mkdir(parents=True, exist_ok=True)

        self.rr = rr
        self.rec = rr.RecordingStream(self.application_id, recording_id=self.recording_id)

        self.rec.save(str(self.rrd_path))

        if self.spawn_viewer:
            self.rec.spawn()

    def __enter__(self) -> "SegmentRecorder":
        self._ctx = self.rec.__enter__()

        self.event(
            f"SEGMENT_START dataset={self.dataset} segment={self.segment} recording_id={self.recording_id}",
            step=0,
        )
        self.segment_meta(self.meta, step=0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc:
            self.event(f"SEGMENT_EXCEPTION {exc_type.__name__}: {exc}", step=0)

        self.event("SEGMENT_END", step=0)
        self.rec.__exit__(exc_type, exc, tb)

        try:
            self.rec.disconnect()
        except Exception:
            pass

    # -------------------------
    # timeline helpers
    # -------------------------

    def set_step(self, step: int) -> None:
        self.rec.set_time("step", sequence=int(step))

    # -------------------------
    # log helpers
    # -------------------------

    def event(self, text: str, *, step: int | None = None) -> None:
        if step is not None:
            self.set_step(step)
        self.rr.log(f"{ROOT}/events", self.rr.TextLog(str(text)))

    def segment_meta(self, meta: dict[str, Any], *, step: int | None = None) -> None:
        if not meta:
            return
        if step is not None:
            self.set_step(step)
        self.rr.log(
            f"{ROOT}/meta/segment",
            self.rr.AnyValues(**{k: str(v) for k, v in meta.items()}),
        )

    def agent_spec(
        self,
        agent: str,
        instance_id: str,
        *,
        profile: str,
        model: str,
        tools_installed: list[str],
        tools_allowed: list[str],
        tools_exposed: list[str],
    ) -> None:
        # static metadata
        self.rr.log(
            f"{ROOT}/meta/agents/{agent}/{instance_id}/spec",
            self.rr.AnyValues(
                agent=agent,
                instance_id=instance_id,
                profile=profile,
                model=model,
                tools_installed=",".join(tools_installed),
                tools_allowed=",".join(tools_allowed),
                tools_exposed=",".join(tools_exposed),
            ),
            static=True,
        )

    def turn(self, agent: str, instance_id: str, text: str, *, step: int) -> None:
        self.set_step(step)
        self.rr.log(
            f"{ROOT}/agents/{agent}/{instance_id}/conversation",
            self.rr.TextLog(f"user: {text}"),
        )

    def assistant(self, agent: str, instance_id: str, text: str, *, step: int) -> None:
        self.set_step(step)
        self.rr.log(
            f"{ROOT}/agents/{agent}/{instance_id}/conversation",
            self.rr.TextLog(f"assistant: {text}"),
        )

    def tool_call(self, agent: str, instance_id: str, name: str, params: dict, *, step: int) -> None:
        self.set_step(step)
        self.rr.log(
            f"{ROOT}/agents/{agent}/{instance_id}/tool_calls",
            self.rr.TextLog(f"{name}({params})"),
        )

    def tool_result(self, agent: str, instance_id: str, name: str, result: str, *, step: int) -> None:
        self.set_step(step)
        self.rr.log(
            f"{ROOT}/agents/{agent}/{instance_id}/tool_results",
            self.rr.TextLog(f"{name} -> {result}"),
        )

    def usage(self, agent: str, instance_id: str, prompt_tokens: int, *, step: int) -> None:
        self.set_step(step)
        self.rr.log(
            f"{ROOT}/agents/{agent}/{instance_id}/usage/prompt_tokens",
            self.rr.Scalars(int(prompt_tokens)),
        )


@dataclass
class SegmentContext:
    """
    Shared per-segment step counter so multiple agents log onto the same timeline.
    """
    recorder: SegmentRecorder
    step_value: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def step(self, delta: int = 1) -> int:
        d = int(delta or 1)
        if d <= 0:
            d = 1
        with self._lock:
            self.step_value += d
            return int(self.step_value)
