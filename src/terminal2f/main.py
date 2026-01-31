import pyarrow as pa
import ulid
import rerun as rr
import rerun.catalog as catalog
import datetime
import time
from pathlib import Path


# config this
EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment, name could even make sense
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}" # thus this is the specfic experiment, and not a dataset more like a workspace


LOGS_DIR = Path("logs")
TABLES_DIR = LOGS_DIR / "tables"
STORAGE_DIR = LOGS_DIR / "storage"  # store recordings here
# ARTIFACTS_DIR = LOGS_DIR / "artifatcs"  # store experiment artifacts here, tool_resuts. bla ba, tables point to this destination.
RECORDINGS = STORAGE_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / "runs"
# in the future RECORDINGS_ROOT = "s3://my-bucket/terminal2f/recordings"

# this is the cli type of work
MODE = "load"  # "record" or "load"
LOAD_RUN_ID = "01KGA8924AFQ16F0XJGQTMSH9T"  # set this when MODE="load", e.g. "01J..."

TASKS = {
    "task_1": {"prompt": "Add 2+2", "expected": "4"},
    "task_2": {"prompt": "Reverse 'abc'", "expected": "cba"},
    "task_3": {"prompt": "Uppercase 'hello'", "expected": "HELLO"},
    "task_4": {"prompt": "Count letters in 'banana'", "expected": "6"},
}

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
        ("layer", pa.string()),  # variant name ("Base"/"A"/"B"/etc)
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
    path = (TABLES_DIR / name).absolute()
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


class Episode:
    # recordings/<family>/<version>/runs/<run_id>/episodes/<episode_id>/<variant>.rrd
    def __init__(self, *, run_id: str, episode_id: str, variants: tuple[str, ...]):
        self.run_id = run_id
        self.episode_id = episode_id
        self.variants = variants
        self.episode_dir = RECORDINGS / run_id / "episodes" / episode_id
        self._paths: dict[str, Path] = {}
        self._recs: dict[str, rr.RecordingStream] = {}

    def __enter__(self):
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        for v in self.variants:
            p = self.episode_dir / f"{v}.rrd"
            rec = rr.RecordingStream(
                application_id=f"{EXPERIMENT_FAMILY}/{VERSION_ID}",
                recording_id=f"{self.run_id}:{self.episode_id}",  # segment id comes from here
            )
            rec.save(str(p))
            self._paths[v] = p
            self._recs[v] = rec

            # Shared properties only (must not conflict across variants/layers).
            with rec:
                # segment-level metadata: should be the same for A and B
                # do I need it who knows.
                rr.send_recording_name(f"{self.run_id}:{self.episode_id}")
                rr.send_property("run_id", rr.AnyValues(value=[self.run_id]))
                rr.send_property("episode_id", rr.AnyValues(value=[self.episode_id]))
        return self

    def recording(self, variant: str) -> rr.RecordingStream:
        return self._recs[variant]

    def prefix(self, variant: str) -> str:
        return f"variants/{variant}"
        # return f"episodes/{self.episode_id}/variants/{variant}"
        # return f"{self.run_id}/variants/{variant}"
        # or not at all

    def __exit__(self, exc_type, exc, tb):
        for rec in self._recs.values():
            rec.flush()
            rec.disconnect()
        return False


def index_episode(
    episodes_table: catalog.TableEntry,
    *,
    run_id: str,
    episode_id: str,
    variants: tuple[str, ...],
    metrics_by_variant: dict[str, tuple[float, int, bool]],
):
    for v in variants:
        total_return, steps, done = metrics_by_variant[v]
        episodes_table.append(
            experiment_family=EXPERIMENT_FAMILY,
            version_id=VERSION_ID,
            run_id=run_id,
            episode_id=episode_id,
            layer=v,
            total_return=float(total_return),
            steps=int(steps),
            done=bool(done),
        )


def load_run_into_dataset(dataset, *, run_id: str):
    run_dir = RECORDINGS / run_id
    for p in sorted(run_dir.rglob("*.rrd")):
        dataset.register(p.absolute().as_uri(), layer_name=p.stem).wait()


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


def run_rl_episode(*, seed: int, horizon: int, policy, root: str) -> tuple[float, int, bool]:
    env = DummyEnv()
    obs = env.reset(seed=seed)

    rr.log(f"{root}/meta/policy", rr.TextLog(getattr(policy, "__name__", "policy")))
    rr.log(f"{root}/meta/seed", rr.Scalars(float(seed)))
    rr.log(f"{root}/meta/horizon", rr.Scalars(float(horizon)))

    total = 0.0
    steps = 0
    done = False

    for step in range(horizon):
        steps = step + 1
        rr.set_time("env_step", sequence=step)

        action = int(policy(obs, step))
        next_obs, reward, done = env.step(action)
        total += reward

        rr.log(f"{root}/obs", rr.Scalars(float(obs)))
        rr.log(f"{root}/action", rr.Scalars(float(action)))
        rr.log(f"{root}/reward", rr.Scalars(float(reward)))
        rr.log(f"{root}/return", rr.Scalars(float(total)))
        rr.log(f"{root}/done", rr.TextLog(str(done)))

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
        run_id = str(ulid.new())
        now = datetime.datetime.now(datetime.timezone.utc)

        # --- A/B benchmark scenario (commented out) ---
        # variants = ("A", "B")
        # for episode_id, spec in TASKS.items():
        #     with Episode(run_id=run_id, episode_id=episode_id, variants=variants) as ep:
        #         for v in variants:
        #             with ep.recording(v):
        #                 root = ep.prefix(v)
        #                 rr.log(f"{root}/task/prompt", rr.TextLog(spec["prompt"]))
        #                 rr.log(f"{root}/task/expected", rr.TextLog(spec["expected"]))
        #                 rr.log(f"{root}/agent/output", rr.TextLog(f"{v} answered something"))
        #     # metrics placeholder for benchmark demo
        #     metrics_by_variant = {v: (0.0, 0, False) for v in variants}
        #     index_episode(episodes_table, run_id=run_id, episode_id=episode_id, variants=variants, metrics_by_variant=metrics_by_variant)

        # --- RL scenario (active): 10 episodes, 2 variants, same seed per-episode ---
        variants = ("A", "B")
        horizon = 12

        for i in range(1, 11):
            episode_id = f"episode_{i}"
            seed = 1000 + i  # deterministic per-episode seed

            metrics_by_variant: dict[str, tuple[float, int, bool]] = {}

            with Episode(run_id=run_id, episode_id=episode_id, variants=variants) as ep:
                for v in variants:
                    with ep.recording(v):
                        metrics_by_variant[v] = run_rl_episode(
                            seed=seed,
                            horizon=horizon,
                            policy=POLICIES[v],
                            root=ep.prefix(v),
                        )

            index_episode(
                episodes_table,
                run_id=run_id,
                episode_id=episode_id,
                variants=variants,
                metrics_by_variant=metrics_by_variant,
            )

        runs_table.append(
            experiment_family=EXPERIMENT_FAMILY,
            version_id=VERSION_ID,
            run_id=run_id,
            start=now,
            end=now,
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
