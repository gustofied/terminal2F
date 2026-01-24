import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr


# -----------------------------
# Naming convention (NO UUIDs)
# -----------------------------
EXPERIMENT = "tool_usage_math"
TRIAL_ID = time.strftime("%Y-%m-%d_run02")  # bump this when you rerun
DATASET_NAME = f"experiments/{EXPERIMENT}/{TRIAL_ID}"

OUT_DIR = Path("rrd_out") / EXPERIMENT / TRIAL_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _sql_safe_name(name: str) -> str:
    """Make a string safe to use as a catalog *table* name (DataFusion SQL identifier)."""
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def get_or_create_dataset(client: rr.catalog.CatalogClient, name: str) -> rr.catalog.DatasetEntry:
    try:
        client.create_dataset(name)
    except rr.catalog.AlreadyExistsError:
        pass
    return client.get_dataset(name=name)


def get_or_create_metrics_table(
    client: rr.catalog.CatalogClient, table_name: str, storage_url: str
) -> rr.catalog.TableEntry:
    schema = pa.schema(
        [
            ("dataset", pa.string()),
            ("segment_id", pa.string()),
            ("variant", pa.string()),
            ("tokens", pa.int64()),
            ("tool_calls", pa.int64()),
            ("steps", pa.int64()),
            ("success", pa.bool_()),
        ]
    )

    try:
        client.create_table(name=table_name, schema=schema, url=storage_url)
    except rr.catalog.AlreadyExistsError:
        pass

    return client.get_table(name=table_name)


def log_variant_rrd(segment_id: str, variant: str) -> str:
    """
    Creates ONE RRD file for (segment_id, variant).

    IMPORTANT:
    - recording_id == segment_id  -> becomes the dataset's `segment_id`
    - variant is stored as the dataset `layer_name` when we register
    """
    rrd_path = OUT_DIR / f"{segment_id}__{variant}.rrd"

    rec = rr.RecordingStream(
        application_id=DATASET_NAME,
        recording_id=segment_id,   # <-- segment_id inside the dataset
    )

    # Write to disk (RRD for dataset ingestion)
    rec.save(str(rrd_path))

    # For dataset-layered A/B, keep entity paths identical across variants
    rec.log("prompt", rr.TextLog(f"solve case={segment_id} variant={variant}"))
    rec.log("answer", rr.TextLog(f"hello from {variant} @ {segment_id}"))
    rec.log("debug/value", rr.Scalars([float(np.random.randn())]))

    return rrd_path.absolute().as_uri()


def live_preview_case(segment_id: str, viewer_started: bool) -> bool:
    """
    Optional: show one "case" as its own live recording in the Viewer.
    This is purely for human eyeballing while the script runs.
    """
    preview = rr.RecordingStream(
        application_id=f"preview/{EXPERIMENT}/{TRIAL_ID}",
        recording_id=segment_id,
    )

    if not viewer_started:
        preview.spawn()          # start Viewer once
        viewer_started = True
    else:
        preview.connect_grpc()   # connect to already running Viewer

    # Here we DO put A/B under different subtrees so you can see both at once
    preview.log("ab/A/answer", rr.TextLog(f"(preview) A answer for {segment_id}"))
    preview.log("ab/B/answer", rr.TextLog(f"(preview) B answer for {segment_id}"))

    return viewer_started


def main():
    viewer_started = False

    # Starts an in-process local dataset server (Redap/OSS server)
    with rr.server.Server() as server:
        client = server.client()

        dataset = get_or_create_dataset(client, DATASET_NAME)

        metrics_table_name = _sql_safe_name(f"leaderboard__{EXPERIMENT}__{TRIAL_ID}")
        metrics_url = (OUT_DIR / "leaderboard.lance").absolute().as_uri()
        metrics_table = get_or_create_metrics_table(client, metrics_table_name, metrics_url)

        rrd_uris: list[str] = []
        rrd_layers: list[str] = []

        for case_idx in range(10):
            segment_id = f"case_{case_idx:04d}"

            # ----- LIVE PREVIEW (segment-by-segment) -----
            viewer_started = live_preview_case(segment_id, viewer_started)

            # ----- WRITE RRDs FOR DATASET -----
            for variant in ["A", "B"]:
                rrd_uri = log_variant_rrd(segment_id, variant)
                rrd_uris.append(rrd_uri)
                rrd_layers.append(variant)

                # Example metrics row (replace with real)
                metrics_table.append(
                    dataset=DATASET_NAME,
                    segment_id=segment_id,
                    variant=variant,
                    tokens=int(np.random.randint(50, 200)),
                    tool_calls=int(np.random.randint(0, 5)),
                    steps=int(np.random.randint(1, 12)),
                    success=bool(np.random.rand() > 0.3),
                )

            time.sleep(0.2)

        # Register recordings into the dataset as A/B layers
        handle = dataset.register(rrd_uris, layer_name=rrd_layers)
        handle.wait()

        addr = server.address()  # e.g. rerun+http://0.0.0.0:57997
        port = addr.rsplit(":", 1)[-1]

        print(f"\n✅ Wrote {len(rrd_uris)} RRDs into dataset={DATASET_NAME}")
        print(f"✅ Leaderboard table name: {metrics_table_name}")
        print(f"Dataset server address: {addr}")
        print(f"\nOpen the dataset in another terminal with:\n  rerun connect 127.0.0.1:{port}\n")

        # Keep the server alive so you can connect to it
        print("Press Ctrl+C to stop the dataset server…")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
