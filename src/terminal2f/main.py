import pyarrow as pa
import ulid
import rerun as rr
import rerun.catalog as catalog
import datetime
import time
from pathlib import Path
import uuid

EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"
LOGS_DIR = Path("logs")
TABLES_DIR = LOGS_DIR / "tables"
STORAGE_DIR = LOGS_DIR / "storage"  # store recordings here
RECORDINGS = STORAGE_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / "runs"

# SUPER TIGHT FLAG:
MODE = "record"  # "record" or "load"
LOAD_RUN_ID = ""  # set this when MODE="load", e.g. "01J..."

# data, data-model, datus

EXPERIMENTS_RUN_SCHEMA: pa.Schema = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("start", pa.timestamp("s", tz="UTC")),
        ("end", pa.timestamp("s", tz="UTC")),
        ("root_rrd_uri", pa.string()),  # REAL POINTER: run root (file://.../runs/<run_id>/)
    ]
)

EXPERIMENTS_EPISODES_SCHEMA: pa.Schema = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("episode_id", pa.string()),
        ("layer", pa.string()),
        ("rrd_uri", pa.string()),  # pointer to the concrete file
    ]
)

# helpies

def init_dataset(client, name: str):
    try:
        client.get_dataset(name=name).delete()
    except LookupError:
        pass
    client.create_dataset(name)
    return client.get_dataset(name=name)


def get_or_make_table(client: catalog.CatalogClient, name: str, schema: pa.Schema) -> catalog.TableEntry:
    path = (TABLES_DIR / name).absolute()
    url = path.as_uri()
    if path.exists():
        client.register_table(name=name, url=url)
    else:
        client.create_table(name=name, schema=schema, url=url)
    return client.get_table(name=name)


# write an episode which the layers will use:
# recordings/<family>/<version>/runs/<run_id>/episodes/<episode_id>/<variant>.rrd
def write_episode_rrd(*, run_id: str, episode_id: str, variant: str) -> str:
    episode_dir = RECORDINGS / run_id / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    rrd_path = episode_dir / f"{variant}.rrd"

    rec = rr.RecordingStream(
        application_id=f"{EXPERIMENT_FAMILY}/{VERSION_ID}",
        recording_id=episode_id,  # segment id comes from here
    )
    rec.save(rrd_path)

    rec.log("prompt", rr.TextLog(f"episode={episode_id} variant={variant}"))
    rec.log("answer", rr.TextLog(f"hello from {variant} @ {episode_id}"))

    return rrd_path.absolute().as_uri()


def load_run_into_dataset(dataset, *, run_id: str):
    run_dir = RECORDINGS / run_id
    episodes_dir = run_dir / "episodes"

    for p in sorted(episodes_dir.rglob("*.rrd")):
        layer = p.stem  # "A" from "A.rrd"
        dataset.register(p.absolute().as_uri(), layer_name=layer).wait()


with rr.server.Server(port=9876) as server:
    client = server.client()

    # dataset is the workspace, it's the current view
    dataset = init_dataset(client, EXPERIMENT)

    # runs table
    runs_table = get_or_make_table(client, "runs", EXPERIMENTS_RUN_SCHEMA)

    # episodes table
    episodes_table = get_or_make_table(client, "episodes", EXPERIMENTS_EPISODES_SCHEMA)

    if MODE == "load":
        run_id = LOAD_RUN_ID
        load_run_into_dataset(dataset, run_id=run_id)

    else:
        run_id = str(ulid.new())
        now = datetime.datetime.now(datetime.timezone.utc)

        # --- Option 1: sequential episodes episode_1..episode_10 (kept for reference) ---
        # for i in range(1, 11):
        #     episode_id = f"episode_{i}"
        #
        #     rrd_uri_a = write_episode_rrd(run_id=run_id, episode_id=episode_id, variant="A")
        #     rrd_uri_b = write_episode_rrd(run_id=run_id, episode_id=episode_id, variant="B")
        #
        #     dataset.register(rrd_uri_a, layer_name="A").wait()
        #     dataset.register(rrd_uri_b, layer_name="B").wait()
        #
        #     episodes_table.append(
        #         experiment_family=EXPERIMENT_FAMILY,
        #         version_id=VERSION_ID,
        #         run_id=run_id,
        #         episode_id=episode_id,
        #         layer="A",
        #         rrd_uri=rrd_uri_a,
        #     )
        #     episodes_table.append(
        #         experiment_family=EXPERIMENT_FAMILY,
        #         version_id=VERSION_ID,
        #         run_id=run_id,
        #         episode_id=episode_id,
        #         layer="B",
        #         rrd_uri=rrd_uri_b,
        #     )

        # --- Option 2: task benchmarking task_1..task_5 with A/B variants (active) ---
        for i in range(1, 6):
            episode_id = f"task_{i}"

            rrd_uri_a = write_episode_rrd(run_id=run_id, episode_id=episode_id, variant="A")
            rrd_uri_b = write_episode_rrd(run_id=run_id, episode_id=episode_id, variant="B")

            dataset.register(rrd_uri_a, layer_name="A").wait()
            dataset.register(rrd_uri_b, layer_name="B").wait()

            episodes_table.append(
                experiment_family=EXPERIMENT_FAMILY,
                version_id=VERSION_ID,
                run_id=run_id,
                episode_id=episode_id,
                layer="A",
                rrd_uri=rrd_uri_a,
            )
            episodes_table.append(
                experiment_family=EXPERIMENT_FAMILY,
                version_id=VERSION_ID,
                run_id=run_id,
                episode_id=episode_id,
                layer="B",
                rrd_uri=rrd_uri_b,
            )

        # REAL POINTER: store run root directory (like W&B run page root)
        run_root_uri = (RECORDINGS / run_id).absolute().as_uri()

        runs_table.append(
            experiment_family=EXPERIMENT_FAMILY,
            version_id=VERSION_ID,
            run_id=run_id,
            start=now,
            end=now,
            root_rrd_uri=run_root_uri,
        )

    print(runs_table.reader())
    print(server.address())
    print(dataset.schema())
    print(episodes_table.reader())

    

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
