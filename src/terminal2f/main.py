from rerun.components import TextLogLevel
import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr


# ------------------- DATA SETUP

EXPERIMENT_NAME = "LOOP_AGENT"
RUN_ID = time.strftime("%Y-%m-%d_rund9")

DATASET_NAME = f"{EXPERIMENT_NAME}/{RUN_ID}"

LOGS_DIR = Path("logs")

RECORDINGS_DIR = LOGS_DIR / "recordings" / EXPERIMENT_NAME / RUN_ID
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR = LOGS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------- METRICS TABLE

METRICS_TABLE_NAME = "eval_metrics"
METRICS_SCHEMA = pa.schema(
    [
        ("experiment", pa.string()),
        ("run_id", pa.string()),
        ("segment_id", pa.string()),
        ("variant", pa.string()),
        ("tokens", pa.int64()),
        ("tool_calls", pa.int64()),
        ("steps", pa.int64()),
        ("success", pa.bool_()),
    ]
)


def log_variant_rrd(segment_id: str, variant: str) -> str:
    rrd_path = RECORDINGS_DIR / f"{segment_id}__{variant}.rrd"

    rec = rr.RecordingStream(
        application_id=DATASET_NAME,
        recording_id=segment_id,
    )
    rec.save(str(rrd_path))

    rec.log("prompt", rr.TextLog(f"solve case={segment_id} variant={variant}", level=TextLogLevel.DEBUG))
    rec.log("answer", rr.TextLog(f"hello from {variant} @ {segment_id}"))
    rec.log("debug/value", rr.Scalars([float(np.random.randn())]))

    return rrd_path.absolute().as_uri()


def live_preview_case(segment_id: str, viewer_started: bool) -> bool:
    preview = rr.RecordingStream(
        application_id=f"preview/{EXPERIMENT_NAME}/{RUN_ID}",
        recording_id=segment_id,
    )

    if not viewer_started:
        preview.spawn()
        viewer_started = True
    else:
        preview.connect_grpc()

    preview.log("ab/A/answer", rr.TextLog(f"(preview) A answer for {segment_id}"))
    preview.log("ab/B/answer", rr.TextLog(f"(preview) B answer for {segment_id}"))

    return viewer_started


def get_or_create_dataset(client: rr.catalog.CatalogClient, name: str) -> rr.catalog.DatasetEntry:
    try:
        client.create_dataset(name)
    except rr.catalog.AlreadyExistsError:
        pass
    return client.get_dataset(name=name)


def ensure_metrics_table(client: rr.catalog.CatalogClient) -> rr.catalog.TableEntry:
    storage_dir = METRICS_DIR / f"{METRICS_TABLE_NAME}.lance"
    storage_url = storage_dir.absolute().as_uri()

    try:
        table = client.get_table(name=METRICS_TABLE_NAME)
    except LookupError:
        table = None

    if table is None:
        if storage_dir.exists():
            client.register_table(name=METRICS_TABLE_NAME, url=storage_url)
        else:
            client.create_table(name=METRICS_TABLE_NAME, schema=METRICS_SCHEMA, url=storage_url)

        table = client.get_table(name=METRICS_TABLE_NAME)

    # Make it queryable in client.ctx
    table.reader()
    return table


def append_metrics_row(
    table: rr.catalog.TableEntry,
    *,
    experiment: str,
    run_id: str,
    segment_id: str,
    variant: str,
    tokens: int,
    tool_calls: int,
    steps: int,
    success: bool,
) -> None:
    table.append(
        experiment=experiment,
        run_id=run_id,
        segment_id=segment_id,
        variant=variant,
        tokens=int(tokens),
        tool_calls=int(tool_calls),
        steps=int(steps),
        success=bool(success),
    )


def run_experiment() -> None:
    viewer_started = False

    with rr.server.Server() as server:
        client = server.client()
        dataset = get_or_create_dataset(client, DATASET_NAME)

        metrics_table = ensure_metrics_table(client)

        rrd_uris: list[str] = []
        rrd_layers: list[str] = []

        for case_idx in range(10):
            segment_id = f"case_{case_idx:04d}"

            viewer_started = live_preview_case(segment_id, viewer_started)

            for variant in ["A", "B"]:
                rrd_uri = log_variant_rrd(segment_id, variant)
                rrd_uris.append(rrd_uri)
                rrd_layers.append(variant)

                append_metrics_row(
                    metrics_table,
                    experiment=EXPERIMENT_NAME,
                    run_id=RUN_ID,
                    segment_id=segment_id,
                    variant=variant,
                    tokens=int(np.random.randint(50, 200)),
                    tool_calls=int(np.random.randint(0, 5)),
                    steps=int(np.random.randint(1, 12)),
                    success=bool(np.random.rand() > 0.3),
                )

            time.sleep(0.2)

        handle = dataset.register(rrd_uris, layer_name=rrd_layers)
        handle.wait()

        # Example query
        batches = client.ctx.sql(f"SELECT * FROM {METRICS_TABLE_NAME} LIMIT 10").collect()
        for b in batches:
            print(b.to_pandas())

        addr = server.address()
        port = addr.rsplit(":", 1)[-1]
        print(f"\n✅ Dataset server address: {addr}")
        print(f"Open Viewer with:\n  rerun connect 127.0.0.1:{port}\n")

        print("Press Ctrl+C to stop…")
        while True:
            time.sleep(1)


def run_sql(sql: str) -> None:
    """
    Lightweight CLI query runner:
    - starts a tiny local server
    - registers/opens the eval_metrics table
    - runs SQL against client.ctx
    """
    with rr.server.Server() as server:
        client = server.client()
        ensure_metrics_table(client)

        batches = client.ctx.sql(sql).collect()
        for b in batches:
            print(b.to_pandas())


def main():
    parser = argparse.ArgumentParser(prog="t2f")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub_sql = sub.add_parser("sql", help="Run a SQL query against the catalog tables")
    sub_sql.add_argument("query", type=str, help='SQL query, e.g. "SELECT * FROM eval_metrics LIMIT 10"')

    args = parser.parse_args()

    if args.cmd == "sql":
        run_sql(args.query)
    else:
        run_experiment()


if __name__ == "__main__":
    main()
