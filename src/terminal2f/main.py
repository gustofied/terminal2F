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
LOAD_RUN_ID = "01KGD9VZCXE1HRJW5D6M091Q1S"  # set this when MODE="load", e.g. "01J..."

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


def index_episode(episodes_table: catalog.TableEntry, *, run_id: str, episode_id: str, layers: tuple[str, ...], metrics_by_layer: dict[str, tuple[float, int, bool]]):
    for l in layers:
        total_return, steps, done = metrics_by_layer[l]
        episodes_table.append(
            experiment_family=EXPERIMENT_FAMILY,
            version_id=VERSION_ID,
            run_id=run_id,
            episode_id=episode_id,
            layer=l,
            total_return=float(total_return),
            steps=int(steps),
            done=bool(done),
        )


def load_run_into_dataset(dataset, *, run_id: str):
    run_dir = RECORDINGS / run_id
    for p in sorted(run_dir.rglob("*.rrd")):
        layer = p.stem  # "A"/"B"
        uri = p.absolute().as_uri()
        dataset.register(uri, layer_name=layer).wait()


# --- RL-style demo runner (env/task owns the actual logging content) ---


class DummyEnv:
    def __init__(self):
        self.state = 0

    def reset(self, *, seed: int) -> int:
        self.state = seed % 10
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        self.state = (self.state * 3 + action) % 10
        reward = float(self.state) / 10.0
        done = self.state == 0
        return self.state, reward, done


def policy_a(obs: int, step: int) -> int:
    return 0


def policy_b(obs: int, step: int) -> int:
    return 1 if (obs + step) % 2 == 0 else 0


POLICIES = {"A": policy_a, "B": policy_b}


def run_rl_episode(*, seed: int, horizon: int, policy, episode: str) -> tuple[float, int, bool]:
    env = DummyEnv()
    obs = env.reset(seed=seed)

    rr.log(f"{episode}/meta/policy", rr.TextLog(getattr(policy, "__name__", "policy")))
    rr.log(f"{episode}/meta/seed", rr.Scalars(float(seed)))
    rr.log(f"{episode}/meta/horizon", rr.Scalars(float(horizon)))

    total = 0.0
    steps = 0
    done = False

    for step in range(horizon):
        steps = step + 1
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
        if done:
            break

    return total, steps, done


with rr.server.Server(port=5555) as server:
    client = server.client()
    dataset = init_dataset(client, EXPERIMENT)

    runs_table = get_or_make_table(client, "runs", EXPERIMENTS_RUN_SCHEMA)
    episodes_table = get_or_make_table(client, "episodes", EXPERIMENTS_EPISODES_SCHEMA)

    if MODE == "load":
        load_run_into_dataset(dataset, run_id=LOAD_RUN_ID)

    else:
        # this is more like the run objet created I suppose
        # could be cleaned up
        run_id = str(ulid.new())
        start = datetime.datetime.now(datetime.timezone.utc)
        run_dir = RECORDINGS / run_id 
        run_dir.mkdir(parents=True, exist_ok=True)
        layers = ("A", "B")
        horizon = 12
        app_id = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"

        for i in range(1, 11):
            episode_id = f"episode_{i}"
            seed = 1000 + i

            metrics_by_layer: dict[str, tuple[float, int, bool]] = {}

            with episode_ctx(
                run_dir=run_dir,
                application_id=app_id,
                run_id=run_id,
                episode_id=episode_id,
                layer="A",
            ) as episode:
                metrics_by_layer["A"] = run_rl_episode(
                    seed=seed,
                    horizon=horizon,
                    policy=policy_a,
                    episode=episode,
                )

            with episode_ctx(
                run_dir=run_dir,
                application_id=app_id,
                run_id=run_id,
                episode_id=episode_id,
                layer="B",
            ) as episode:
                metrics_by_layer["B"] = run_rl_episode(
                    seed=seed,
                    horizon=horizon,
                    policy=policy_b,
                    episode=episode,
                )

            index_episode(
                episodes_table,
                run_id=run_id,
                episode_id=episode_id,
                layers=layers,
                metrics_by_layer=metrics_by_layer,
            )

        end = datetime.datetime.now(datetime.timezone.utc)

        runs_table.append(
            experiment_family=EXPERIMENT_FAMILY,
            version_id=VERSION_ID,
            run_id=run_id,
            start=start,
            end=end,
        )

    print(runs_table.reader())
    print(server.url())
    print(dataset.schema())
    print(episodes_table.reader())
    print(client.url)
    print(client.entries())

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
