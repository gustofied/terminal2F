# can try to get this to a simple 400-500 loc, which kinda goes through all and recordsa an A/B rollout perhaps??

import pyarrow as pa
import ulid
import rerun as rr
import rerun.catalog as catalog
import datetime
import time
from pathlib import Path
from contextlib import contextmanager

# config this
EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment, name could even make sense
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"  # thus this is the specfic experiment, and not a dataset more like a workspace, rename to worksapce??

LOGS_DIR = Path("logs")
TABLES_DIR = LOGS_DIR / "tables"
STORAGE_DIR = LOGS_DIR / "storage"  # store recordings here
RECORDINGS = STORAGE_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / "runs"
# in the future RECORDINGS_ROOT = "s3://my-bucket/terminal2f/recordings"

# this is the cli type of work
MODE = "record"  # "record" or "load"
LOAD_RUN_ID = ""  # set this when MODE="load", e.g. "01J..."

EXPERIMENTS_RUN_SCHEMA: pa.Schema = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("start", pa.timestamp("s", tz="UTC")),
        ("end", pa.timestamp("s", tz="UTC")),
    ]
)

EXPERIMENTS_EPISODES_SCHEMA: pa.Schema = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("episode_id", pa.string()),
        ("layer", pa.string()),  # variant name ("A"/"B"/etc)
        ("total_return", pa.float64()),
        ("steps", pa.int64()),
        ("done", pa.bool_()),
    ]
)

def init_dataset(client, name: str):
    try:
        client.get_dataset(name=name).delete()
    except LookupError:
        pass
    client.create_dataset(name)
    return client.get_dataset(name=name)


def get_or_make_table(client: catalog.CatalogClient, name: str, schema: pa.Schema) -> catalog.TableEntry:
    path = (TABLES_DIR / EXPERIMENT_FAMILY / VERSION_ID / name).absolute()
    path.mkdir(parents=True, exist_ok=True)
    url = path.as_uri()

    try:
        return client.get_table(name=name)
    except LookupError:
        if any(path.iterdir()):
            client.register_table(name=name, url=url)
        else:
            client.create_table(name=name, schema=schema, url=url)
        return client.get_table(name=name)

class Policy:
    def __init__(self, name: str, act):
        self.name = name
        self.act = act

    def __call__(self, obs: int, step: int) -> int:
        return self.act(obs, step)


def load_run_into_dataset(dataset, *, run_id: str, policies: list[Policy]):
    run_dir = RECORDINGS / run_id
    for policy in policies:
        prefix = (run_dir / policy.name).absolute().as_uri()
        dataset.register_prefix(prefix, layer_name=policy.name).wait()


class Run:
    def __init__(
        self,
        *,
        experiment_family: str,
        version_id: str,
        recordings_root: Path,
        runs_table: catalog.TableEntry,
        episodes_table: catalog.TableEntry,
        policies: list[Policy],
        num_episodes: int,
        seed_offset: int = 1000,
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
        self.seed_offset = seed_offset
        self.start: datetime.datetime | None = None
        self.end: datetime.datetime | None = None

    def __enter__(self):
        self.start = datetime.datetime.now(datetime.timezone.utc)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self

    # TODO: __iter__ could yield a Task dataclass (episode_id, seed, policy, prompt, ground_truth, etc.) when real benchmark tasks define the shape
    def __iter__(self):
        for i in range(1, self.num_episodes + 1):
            for policy in self.policies:
                yield f"episode_{i}", self.seed_offset + i, policy

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
        # rec.connect_grpc() not working as i want..
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


# --- ENVyus

class DummyEnv:
    def __init__(self, horizon: int = 5):
        self.horizon = horizon
        self._step = 0

    def reset(self) -> int:
        self._step = 0
        return self._step

    def step(self, action: int) -> tuple[int, float, bool]:
        self._step += 1
        done = self._step >= self.horizon
        return self._step, 1.0, done


# --- Policies

def act_baseline(obs, step):
    return 0

def act_alternating(obs, step):
    return 1 if (obs + step) % 2 == 0 else 0

POLICIES = [
    Policy("baseline", act_baseline),
    Policy("alternating", act_alternating),
]

def rollout(*, seed: int, policy, episode: str) -> tuple[float, int, bool]:
    env = DummyEnv()
    obs = env.reset()

    rr.log(f"{episode}/meta/policy", rr.TextLog(policy.name))

    total = 0.0
    step = 0
    done = False

    while not done:
        rr.set_time("env_step", sequence=step)

        action = int(policy(obs, step))
        next_obs, reward, done = env.step(action)
        total += reward

        rr.log(f"{episode}/obs", rr.Scalars(float(obs)))
        rr.log(f"{episode}/action", rr.Scalars(float(action)))
        rr.log(f"{episode}/reward", rr.Scalars(float(reward)))
        rr.log(f"{episode}/return", rr.Scalars(float(total)))
        rr.log(f"{episode}/done", rr.TextLog(str(done)))

        obs = next_obs
        step += 1

    return total, step, done


with rr.server.Server(port=5555) as server:
    client = server.client()
    dataset = init_dataset(client, EXPERIMENT)
    runs_table = get_or_make_table(client, "runs", EXPERIMENTS_RUN_SCHEMA)
    episodes_table = get_or_make_table(client, "episodes", EXPERIMENTS_EPISODES_SCHEMA)
    if MODE == "load":
        load_run_into_dataset(dataset, run_id=LOAD_RUN_ID, policies=POLICIES)
    else:
        with Run(experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID, recordings_root=RECORDINGS, runs_table=runs_table, episodes_table=episodes_table, policies=POLICIES, num_episodes=10) as run:
            for episode_id, seed, policy in run:   # TODO: __iter__ could yield a Task dataclass (episode_id, seed, policy, prompt, ground_truth, etc.) when real benchmark tasks define the shape
                with run.episode(episode_id, layer=policy.name) as episode:
                    total_return, steps, done = rollout(seed=seed, policy=policy, episode=episode)
                    run.log_metrics(episode_id=episode_id, layer=policy.name, total_return=total_return, steps=steps, done=done)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
