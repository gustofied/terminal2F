import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr


# -----------------------------
# Experiment naming
# -----------------------------
EXPERIMENT = "tool_usage_math"
TRIAL_ID = time.strftime("%Y-%m-%d_run06")
DATASET_NAME = f"experiments/{EXPERIMENT}/{TRIAL_ID}"

OUT_DIR = Path("rrd_out") / EXPERIMENT / TRIAL_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path("logs")  # where the leaderboard Lance dataset lives


# -----------------------------
# Catalog table schema (leaderboard)
# -----------------------------
LEADERBOARD_TABLE_NAME = "leaderboard"
LEADERBOARD_SCHEMA = pa.schema(
    [
        ("experiment", pa.string()),
        ("trial_id", pa.string()),
        ("segment_id", pa.string()),
        ("variant", pa.string()),
        ("tokens", pa.int64()),
        ("tool_calls", pa.int64()),
        ("steps", pa.int64()),
        ("success", pa.bool_()),
    ]
)


def _as_uri(path: Path) -> str:
    return path.absolute().as_uri()


# -----------------------------
# RRD logging
# -----------------------------
def log_variant_rrd(segment_id: str, variant: str) -> str:
    """
    Creates ONE RRD file for (segment_id, variant).
    Uses the same entity paths across variants so layered A/B compares cleanly.
    """
    rrd_path = OUT_DIR / f"{segment_id}__{variant}.rrd"

    rec = rr.RecordingStream(
        application_id=DATASET_NAME,
        recording_id=segment_id,
    )

    rec.save(str(rrd_path))

    rec.log("prompt", rr.TextLog(f"solve case={segment_id} variant={variant}"))
    rec.log("answer", rr.TextLog(f"hello from {variant} @ {segment_id}"))
    rec.log("debug/value", rr.Scalars([float(np.random.randn())]))

    return rrd_path.absolute().as_uri()


def live_preview_case(segment_id: str, viewer_started: bool) -> bool:
    """
    Optional: show one "case" live in the Viewer while the script runs.
    """
    preview = rr.RecordingStream(
        application_id=f"preview/{EXPERIMENT}/{TRIAL_ID}",
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


# -----------------------------
# Catalog helpers
# -----------------------------
def get_or_create_dataset(client: rr.catalog.CatalogClient, name: str) -> rr.catalog.DatasetEntry:
    try:
        client.create_dataset(name)
    except rr.catalog.AlreadyExistsError:
        pass
    return client.get_dataset(name=name)


def ensure_leaderboard_table(client: rr.catalog.CatalogClient, root: Path) -> rr.catalog.TableEntry:
    """
    Ensure the leaderboard catalog table exists and is queryable via client.ctx.

    Rerun 0.28.x behavior:
    - create_table(...) creates a NEW Lance dataset directory at the url. :contentReference[oaicite:5]{index=5}
    - if the Lance dir already exists, you must register_table(...) instead. :contentReference[oaicite:6]{index=6}
    - get_table(name=...) raises LookupError if not found (observed in practice).
    - TableEntry.reader() registers the table with DataFusion for SQL queries. :contentReference[oaicite:7]{index=7}
    """
    storage_dir = root / "metrics" / f"{LEADERBOARD_TABLE_NAME}.lance"
    storage_dir.parent.mkdir(parents=True, exist_ok=True)
    storage_url = _as_uri(storage_dir)

    # Try fetch from catalog
    try:
        table = client.get_table(name=LEADERBOARD_TABLE_NAME)
    except LookupError:
        table = None

    if table is None:
        # If disk already has a Lance dataset, register it; else create it
        if storage_dir.exists():
            client.register_table(name=LEADERBOARD_TABLE_NAME, url=storage_url)  # :contentReference[oaicite:8]{index=8}
        else:
            client.create_table(  # :contentReference[oaicite:9]{index=9}
                name=LEADERBOARD_TABLE_NAME,
                schema=LEADERBOARD_SCHEMA,
                url=storage_url,
            )

        table = client.get_table(name=LEADERBOARD_TABLE_NAME)

    # Make it queryable in client.ctx
    table.reader()  # :contentReference[oaicite:10]{index=10}
    return table


def append_leaderboard_row(
    table: rr.catalog.TableEntry,
    *,
    experiment: str,
    trial_id: str,
    segment_id: str,
    variant: str,
    tokens: int,
    tool_calls: int,
    steps: int,
    success: bool,
) -> None:
    """
    Append one row to the leaderboard using TableEntry.append. :contentReference[oaicite:11]{index=11}
    """
    table.append(  # :contentReference[oaicite:12]{index=12}
        experiment=experiment,
        trial_id=trial_id,
        segment_id=segment_id,
        variant=variant,
        tokens=int(tokens),
        tool_calls=int(tool_calls),
        steps=int(steps),
        success=bool(success),
    )


# -----------------------------
# Main
# -----------------------------
def main():
    viewer_started = False

    with rr.server.Server() as server:
        client = server.client()

        dataset = get_or_create_dataset(client, DATASET_NAME)

        leaderboard = ensure_leaderboard_table(client, LOGS_DIR)

        rrd_uris: list[str] = []
        rrd_layers: list[str] = []

        for case_idx in range(10):
            segment_id = f"case_{case_idx:04d}"

            viewer_started = live_preview_case(segment_id, viewer_started)

            for variant in ["A", "B"]:
                rrd_uri = log_variant_rrd(segment_id, variant)
                rrd_uris.append(rrd_uri)
                rrd_layers.append(variant)

                append_leaderboard_row(
                    leaderboard,
                    experiment=EXPERIMENT,
                    trial_id=TRIAL_ID,
                    segment_id=segment_id,
                    variant=variant,
                    tokens=int(np.random.randint(50, 200)),
                    tool_calls=int(np.random.randint(0, 5)),
                    steps=int(np.random.randint(1, 12)),
                    success=bool(np.random.rand() > 0.3),
                )

            time.sleep(0.2)

        # Register RRDs as layers (supports list URIs + list layer names). :contentReference[oaicite:13]{index=13}
        handle = dataset.register(rrd_uris, layer_name=rrd_layers)  # :contentReference[oaicite:14]{index=14}
        handle.wait()

        # Query via catalog-owned DataFusion context. :contentReference[oaicite:15]{index=15}
        ctx = client.ctx  # :contentReference[oaicite:16]{index=16}

        result_batches = ctx.sql(
            f"""
            SELECT
              experiment,
              trial_id,
              variant,
              COUNT(*) AS n,
              AVG(CAST(success AS DOUBLE)) AS success_rate,
              AVG(tokens) AS avg_tokens,
              AVG(tool_calls) AS avg_tool_calls
            FROM {LEADERBOARD_TABLE_NAME}
            WHERE experiment = '{EXPERIMENT}' AND trial_id = '{TRIAL_ID}'
            GROUP BY experiment, trial_id, variant
            ORDER BY variant
            """
        ).collect()

        print("\n=== Leaderboard summary ===")
        for batch in result_batches:
            print(batch.to_pandas())

        addr = server.address()
        port = addr.rsplit(":", 1)[-1]
        print(f"\n✅ Dataset server address: {addr}")
        print(f"Open with:\n  rerun connect 127.0.0.1:{port}\n")

        print("Press Ctrl+C to stop…")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
