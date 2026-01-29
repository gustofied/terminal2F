import pyarrow as pa
import pyarrow.parquet as pq
import ulid
import rerun as rr
import rerun.catalog as catalog 
import datetime
import time
from pathlib import Path
import uuid

EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"  # stable dataset name¨
LOGS_DIR = Path("logs")
TABLES_DIR = LOGS_DIR / "tables"  # stable across all runs
STORAGE_DIR = LOGS_DIR / "storage"

# data, data-model, datus

EXPERIMENTS_RUN_SCHEMA: pa.Schema = pa.schema([
    ("experiment_family", pa.string()),
    ("version_id", pa.string()),
    ("run_id", pa.string()),
    ("start", pa.timestamp("s", tz="UTC")),
    ("end", pa.timestamp("s", tz="UTC")),
    ("rrd_uri_a", pa.string()),
    ("rrd_uri_b", pa.string()),
])

EXPERIMENTS_METRICS_SCHEMA: pa.Schema = pa.schema(

    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("suite_name", pa.string()),
        ("task_id", pa.string()),  # stable task key within suite
        ("trial_id", pa.string()),  # shared across A/B
        ("episode_id", pa.string()),  # unique per (trial, variant)
        ("problem_id", pa.string()),  # stable problem id / task / rollout (suite/task)
        ("variant", pa.string()),  # A/B
        ("rrd_uri", pa.string()),  # rrd rerun recording pointer
        ("tokens", pa.int64()),
        ("success", pa.bool_()),
        ("wall_time_ms", pa.int64()),
    ]

)
# helpies

def reset_dataset(client, name: str):
    try:
        client.get_dataset(name=name).delete()
    except LookupError:
        pass
    client.create_dataset(name)
    return client.get_dataset(name=name)

def make_table(client: catalog.CatalogClient, name: str, schema: pa.Schema, path: Path) -> catalog.TableEntry:
    path = path.absolute()
    url = path.as_uri()
    if path.exists():
        client.register_table(name=name, url=url)
    else:
        client.create_table(name=name, schema=schema, url=url)
    return client.get_table(name=name)

#write an eppsiode which the layers will use / epsiode / variants
def write_episode_rrd(recordings_dir: Path, *, problem_id: str, episode_id: str, variant: str) -> str:
    rrd_path = recordings_dir / problem_id / f"{episode_id}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rec = rr.RecordingStream(
        application_id=f"{EXPERIMENT_FAMILY}/{VERSION_ID}",
        recording_id=problem_id,  # segment id
    )
    rec.save(str(rrd_path))

    rec.log("prompt", rr.TextLog(f"solve problem={problem_id} episode={episode_id} variant={variant}"))
    rec.log("answer", rr.TextLog(f"hello from {variant} @ {problem_id} (episode={episode_id})"))

    return rrd_path.absolute().as_uri()


with rr.server.Server(port=9876) as server:
    print(server.address())
    client = server.client()

    dataset = reset_dataset(client, EXPERIMENT)
    
    runs_table = make_table(client, "experiment_run", EXPERIMENTS_RUN_SCHEMA, STORAGE_DIR)

    run_id = str(ulid.new())
    now = datetime.datetime.now(datetime.timezone.utc)

    recordings_dir = LOGS_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / run_id
    recordings_dir.mkdir(parents=True, exist_ok=True)

    # one segment per run (just for this test) #test/suite
    problem_id = f"layer_test/{run_id}"

    rrd_uri_a = write_episode_rrd(recordings_dir, problem_id=problem_id, episode_id=str(uuid.uuid4()), variant="A")
    rrd_uri_b = write_episode_rrd(recordings_dir, problem_id=problem_id, episode_id=str(uuid.uuid4()), variant="B")

    # dataset shows two layers for the same segment/problem_id
    dataset.register(rrd_uri_a, layer_name="A").wait()
    dataset.register(rrd_uri_b, layer_name="B").wait()

    runs_table.append(
        experiment_family=EXPERIMENT_FAMILY,
        version_id=VERSION_ID,
        run_id=run_id,
        start=now,
        end=now,
        rrd_uri_a=rrd_uri_a,
        rrd_uri_b=rrd_uri_b,
    )

    print(runs_table.reader())
    print(server.address())
    print(rrd_uri_a)  
    print(dataset.schema())


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass



