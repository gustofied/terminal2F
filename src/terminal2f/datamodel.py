from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import rerun.catalog as catalog


# Schema

RUNS_SCHEMA: pa.Schema = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("start", pa.timestamp("s", tz="UTC")),
        ("end", pa.timestamp("s", tz="UTC")),
    ]
)

EPISODES_SCHEMA: pa.Schema = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("episode_id", pa.string()),
        ("layer", pa.string()),
        ("total_return", pa.float64()),
        ("steps", pa.int64()),
        ("done", pa.bool_()),
        ("final_confidence", pa.float64()),
        ("final_entropy", pa.float64()),
        ("total_cost", pa.float64()),
        ("true_hypothesis", pa.string()),
        ("chosen_hypothesis", pa.string()),
        ("correct", pa.bool_()),
    ]
)

# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
TABLES_DIR = LOGS_DIR / "tables"
STORAGE_DIR = LOGS_DIR / "storage"
# in the future RECORDINGS_ROOT = "s3://my-bucket/terminal2f/recordings"


def recordings_path(experiment_family: str, version_id: str) -> Path:
    return STORAGE_DIR / "recordings" / experiment_family / version_id / "runs"


# Catalog

def init_dataset(client, name: str):
    try:
        client.get_dataset(name=name).delete()
    except LookupError:
        pass
    client.create_dataset(name)
    return client.get_dataset(name=name)


def get_or_make_table(
    client: catalog.CatalogClient,
    name: str,
    schema: pa.Schema,
    *,
    experiment_family: str,
    version_id: str,
) -> catalog.TableEntry:
    path = (TABLES_DIR / experiment_family / version_id / name).absolute()
    path.mkdir(parents=True, exist_ok=True)
    url = path.as_uri()

    try:
        return client.get_table(name=name)
    except LookupError:
        subdirs = [d for d in path.iterdir() if d.is_dir()] if path.exists() else []
        if subdirs:
            # create_table stores lance data in a UUID subdirectory
            lance_url = subdirs[0].absolute().as_uri()
            client.register_table(name=name, url=lance_url)
        else:
            client.create_table(name=name, schema=schema, url=url)
        return client.get_table(name=name)


def load_run_into_dataset(dataset, *, run_id: str, recordings: Path, policies: list):
    run_dir = recordings / run_id
    for policy in policies:
        prefix = (run_dir / policy.name).absolute().as_uri()
        dataset.register_prefix(prefix, layer_name=policy.name).wait()
