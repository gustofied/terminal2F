import pyarrow as pa
import pyarrow.parquet as pq
import ulid
import rerun as rr
import rerun.catalog as Catalog # maybe lower casey
import datetime
import time
from pathlib import Path
import uuid

# days = pa.array([1, 12, 17, 23, 28], type=pa.int8())
# months = pa.array([1, 2, 3, 5, 7], type=pa.int8())
# years = pa.array([1990, 2000, 1995, 2000, 1995], type=pa.int16())
# birthdays_table = pa.table(
#     [days, months, years],
#     ["days", "months", "years"])
# print(birthdays_table)

# pq.write_table(birthdays_table, "learning.paquet")

# print("- - - - - ")

# arr = pa.array([1, 2, 3, 4, 6])
# print(arr.type)

# arr = arr.cast(pa.int8())
# print(arr.type)

# schema = pa.schema([
#     ("col1", pa.int8()),
#     ("col2", pa.string()),
#     ("col3", pa.float64())
# ])

# print(schema)

# table = pa.table([
#     [1, 4, 6],
#     ["a", "d", "f"], 
#     [4.30, 3.30, 4.00]
# ],schema=schema)

# print(table)

# data, data-model, datus

storage_path = Path("learning.lance").absolute()
storage_url = storage_path.as_uri()


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

def make_run_id() -> str: 
    return str(ulid.new()) # unsure if ulid is what I should do here but aight for now


now = datetime.datetime.now(datetime.timezone.utc)

# run_example_table: pa.Table = pa.table([
#     ["tools", "no-tools", "my-tools"],
#     ["v1", "v1", "v1"],
#     [make_run_id(), make_run_id(), make_run_id()],
#     [now, now, now], 
#     [now, now, now], 
# ], schema=EXPERIMENTS_RUN_SCHEMA)


def reset_dataset(client, name: str):
    try:
        client.get_dataset(name=name).delete()
    except LookupError:
        pass
    client.create_dataset(name)
    return client.get_dataset(name=name)

def make_table(client: Catalog.CatalogClient, name: str, schema: pa.Schema, path: Path) -> Catalog.TableEntry:
    path = path.absolute()
    url = path.as_uri()
    if path.exists():
        client.register_table(name=name, url=url)
    else:
        client.create_table(name=name, schema=schema, url=url)
    return client.get_table(name=name)



EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment

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


EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"  # stable dataset name
LOGS_DIR = Path("logs")
TABLES_DIR = LOGS_DIR / "tables"  # stable across all runs
run_id = make_run_id()
recordings_dir = LOGS_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / run_id
recordings_dir.mkdir(parents=True, exist_ok=True)

with rr.server.Server() as server:
    print(server.address())
    client = server.client()

    dataset = reset_dataset(client, EXPERIMENT)

    runs_table = make_table(client, "experiment_run", EXPERIMENTS_RUN_SCHEMA, storage_path)

    run_id = make_run_id()
    now = datetime.datetime.now(datetime.timezone.utc)

    recordings_dir = LOGS_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / run_id
    recordings_dir.mkdir(parents=True, exist_ok=True)

    # one segment per run (just for this test)
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

    # for b in client.ctx.sql("SELECT * FROM experiment_run").collect():
    #     print(b.to_pandas())

        
    print(server.address())


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass



