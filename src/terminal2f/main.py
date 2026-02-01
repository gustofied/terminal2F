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
MODE = "load"  # "record" or "load"
LOAD_RUN_ID = "01KGDE1QMAVDT5DDKDQKYZMQBF"  # set this when MODE="load", e.g. "01J..."

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


def load_run_into_dataset(dataset, *, run_id: str):
    run_dir = RECORDINGS / run_id
    for p in sorted(run_dir.rglob("*.rrd")):
        layer = p.stem  # "A"/"B"
        uri = p.absolute().as_uri()
        dataset.register(uri, layer_name=layer).wait()

    
class Run:
    def __init__(
        self,
        *,
        experiment_family: str,
        version_id: str,
        recordings_root: Path,
        runs_table: catalog.TableEntry,
        episodes_table: catalog.TableEntry,
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
        self.num_episodes = num_episodes
        self.seed_offset = seed_offset
        self.start: datetime.datetime | None = None
        self.end: datetime.datetime | None = None

    def __enter__(self):
        self.start = datetime.datetime.now(datetime.timezone.utc)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __iter__(self):
        for i in range(1, self.num_episodes + 1):
            yield f"episode_{i}", self.seed_offset + i

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
def episode_ctx(*, run_dir: Path, application_id: str, run_id: str, episode_id: str, layer: str):
    # TODO: when moving to async, yield rec explicitly and use rec.log() instead of rr.log() to avoid thread-local conflicts 
    """
    Creates and binds a recording:
      recordings/<family>/<version>/runs/<run_id>/episodes/<episode_id>/<layer>.rrd
    While inside the context, rr.log(...) writes into this .rrd.
    """
    episode_dir = run_dir / "episodes" / episode_id  
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / f"{layer}.rrd"
    rec = rr.RecordingStream(
        application_id=application_id, # Experiment
        recording_id=f"{run_id}:{episode_id}",  # segment = task instance, same across variants
    )
    rec.save(str(path))
    rr.set_thread_local_data_recording(rec)

    try:
        # invariant metadata (is identical across layers)
        rr.send_recording_name(f"{run_id}:{episode_id}")
        rr.send_property("run_id", rr.AnyValues(value=[run_id]))
        rr.send_property("episode_id", rr.AnyValues(value=[episode_id]))

        yield f"episodes/{episode_id}"
    finally:
        rec.flush()
        rec.disconnect()
        rr.set_thread_local_data_recording(None)


# --- RL-style demo runner (env/task owns the actual logging content) ---


class DummyEnv:
    def __init__(self, horizon: int = 12):
        self.state = 0
        self.horizon = horizon

    def reset(self, *, seed: int) -> int:
        self.state = seed % 10
        self._step = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        self.state = (self.state * 3 + action) % 10
        self._step += 1
        reward = float(self.state) / 10.0
        done = self.state == 0 or self._step >= self.horizon
        return self.state, reward, done


class Policy:
    def __init__(self, name: str, fn):
        self.name = name
        self.fn = fn

    def __call__(self, obs: int, step: int) -> int:
        return self.fn(obs, step)


POLICIES = [
    Policy("baseline", lambda obs, step: 0),
    Policy("alternating", lambda obs, step: 1 if (obs + step) % 2 == 0 else 0),
]

def run_rl_episode(*, seed: int, policy, episode: str) -> tuple[float, int, bool]:
    env = DummyEnv()
    obs = env.reset(seed=seed)

    rr.log(f"{episode}/meta/policy", rr.TextLog(policy.name))
    rr.log(f"{episode}/meta/seed", rr.Scalars(float(seed)))

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
        load_run_into_dataset(dataset, run_id=LOAD_RUN_ID)
    else:
        with Run(experiment_family=EXPERIMENT_FAMILY, version_id=VERSION_ID, recordings_root=RECORDINGS, runs_table=runs_table, episodes_table=episodes_table, num_episodes=10) as run:
            for episode_id, seed in run:
                for policy in POLICIES:
                    with episode_ctx(run_dir=run.run_dir, application_id=run.app_id, run_id=run.run_id, episode_id=episode_id, layer=policy.name) as episode:
                        total_return, steps, done = run_rl_episode(seed=seed, policy=policy, episode=episode)
                        run.log_metrics(episode_id=episode_id, layer=policy.name, total_return=total_return, steps=steps, done=done)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
