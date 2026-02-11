from __future__ import annotations

import datetime
from contextlib import contextmanager
from pathlib import Path

import rerun as rr
import rerun.catalog as catalog
import ulid


# Episodic memory lives here — at the Run/episode level, not inside FSM/PDA/TM.
# Within-episode computation uses Memory (messages, stack, etc).
# Across-episode learning uses the recordings, tables, and metrics captured here.
# The episode context manager is the natural boundary for an "episode" in the episodic memory sense.
class Run:
    def __init__(
        self,
        *,
        experiment_family: str,
        version_id: str,
        recordings_root: Path,
        runs_table: catalog.TableEntry,
        episodes_table: catalog.TableEntry,
        policies: list,
        num_episodes: int,
    ):
        self.experiment_family = experiment_family
        self.version_id = version_id
        self.run_id = str(ulid.new())
        self.run_dir = recordings_root / self.run_id
        self.app_id = f"{experiment_family}/{version_id}"
        self.runs_table = runs_table
        self.episodes_table = episodes_table
        self.policies = policies
        self.num_episodes = num_episodes
        self.start: datetime.datetime | None = None
        self.end: datetime.datetime | None = None

    def __enter__(self):
        self.start = datetime.datetime.now(datetime.timezone.utc)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self

    # TODO: __iter__ could yield a Task dataclass (episode_id, seed, policy, prompt, ground_truth, etc.) when real benchmark tasks define the shape
    # a bit bit uggly
    def __iter__(self):
        for i in range(1, self.num_episodes + 1):
            for policy in self.policies:
                yield f"episode_{i}", policy

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = datetime.datetime.now(datetime.timezone.utc)
        self.runs_table.append(
            experiment_family=self.experiment_family,
            version_id=self.version_id,
            run_id=self.run_id,
            start=self.start,
            end=self.end,
        )

    def log_metrics(self, *, episode_id: str, layer: str, total_return: float, steps: int, done: bool):
        self.episodes_table.append(
            experiment_family=self.experiment_family,
            version_id=self.version_id,
            run_id=self.run_id,
            episode_id=episode_id,
            layer=layer,
            total_return=float(total_return),
            steps=int(steps),
            done=bool(done),
        )

    @contextmanager
    def episode(self, episode_id: str, *, layer: str):
        # TODO: when moving to async, yield rec explicitly and use rec.log() instead of rr.log() to avoid thread-local conflicts
        policy_dir = self.run_dir / layer
        policy_dir.mkdir(parents=True, exist_ok=True)
        path = policy_dir / f"{episode_id}.rrd"
        rec = rr.RecordingStream(
            application_id=self.app_id,
            recording_id=f"{self.run_id}:{episode_id}",
        )
        rec.save(str(path))
        rr.set_thread_local_data_recording(rec)

        try:
            rr.send_recording_name(f"{self.run_id}:{episode_id}")
            rr.send_property("run_id", rr.AnyValues(value=[self.run_id]))
            rr.send_property("episode_id", rr.AnyValues(value=[episode_id]))

            yield f"episodes/{episode_id}"
        finally:
            rec.flush() # edge case if it throws but who cares
            rec.disconnect()
            rr.set_thread_local_data_recording(None)
