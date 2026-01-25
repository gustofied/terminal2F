import time
from pathlib import Path

import numpy as np
import rerun as rr

from terminal2f.infra.query_engine import make_ctx
from terminal2f.infra.metrics_store import MetricsStore


EXPERIMENT = "tool_usage_math"
TRIAL_ID = time.strftime("%Y-%m-%d_run05")
DATASET_NAME = f"experiments/{EXPERIMENT}/{TRIAL_ID}"

OUT_DIR = Path("rrd_out") / EXPERIMENT / TRIAL_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path("logs")  # your app-owned metrics storage root


def log_variant_rrd(segment_id: str, variant: str) -> str:
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


def main():
    viewer_started = False

    # ---- YOUR ENGINE ----
    ctx = make_ctx()
    metrics = MetricsStore(root=LOGS_DIR, ctx=ctx)

    # ---- RERUN DATASET SERVER (optional but nice UX) ----
    with rr.server.Server() as server:
        client = server.client()
        dataset = client.create_dataset(DATASET_NAME)  # newer API sometimes exists

        rrd_uris: list[str] = []
        rrd_layers: list[str] = []

        for case_idx in range(10):
            segment_id = f"case_{case_idx:04d}"

            viewer_started = live_preview_case(segment_id, viewer_started)

            for variant in ["A", "B"]:
                rrd_uri = log_variant_rrd(segment_id, variant)
                rrd_uris.append(rrd_uri)
                rrd_layers.append(variant)

                # ---- APPEND METRICS TO YOUR OWN TABLE ----
                metrics.append_row(
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

        # Register recordings into rerun dataset as layers
        handle = dataset.register(rrd_uris, layer_name=rrd_layers)
        handle.wait()

        # ---- QUERY YOUR METRICS WITH DATAFUSION ----
        metrics.register_in_datafusion()

        # Example: leaderboard summary
        result = ctx.sql(
            f"""
            SELECT
              experiment,
              trial_id,
              variant,
              COUNT(*) AS n,
              AVG(CAST(success AS DOUBLE)) AS success_rate,
              AVG(tokens) AS avg_tokens,
              AVG(tool_calls) AS avg_tool_calls
            FROM {metrics.table_name}
            WHERE experiment = '{EXPERIMENT}' AND trial_id = '{TRIAL_ID}'
            GROUP BY experiment, trial_id, variant
            ORDER BY variant
            """
        ).collect()

        print("\n=== Leaderboard summary ===")
        for batch in result:
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
