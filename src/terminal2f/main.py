import argparse  # rather type
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.catalog as catalog  # Catalog SDK
from datetime import datetime, timezone

# ------------------- DATA SETUP

EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"  # (this is *The* Experiment you are doing) The experiment is dervied from famliy and version
RUN_ID = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # A run of the expeiment, unqiue at time.. (can tighten up this at some point)

LOGS_DIR = Path("logs")

# replay of all unqiue episode, per agent, per task, per rollout etc..
# replay / debug
# same blueprint per dataset..
RECORDINGS_DIR = LOGS_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / RUN_ID
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# tables for anaytlics, metrics, evals, training..
# lance tables..
TABLES_DIR = LOGS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- DATA MODEL

RUNS_TABLE_NAME = "runs"  # rename or better later to clear le mental model
EPISODE_METRICS_TABLE_NAME = "episode_metrics"

RUNS_SCHEMA = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("started_at_unix_s", pa.float64()),
        ("ended_at_unix_s", pa.float64()),
    ]
)

EPISODE_METRIC_SCHEMA = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("suite_name", pa.string()),
        ("task_id", pa.string()),  # episode id, fixy later
        ("segment_id", pa.string()),
        ("layer", pa.string()),
        ("rrd_uri", pa.string()),
        ("tokens", pa.int64()),
        ("success", pa.bool_()),
        ("wall_time_ms", pa.int64()),
    ]
)

# ------------------- HELPERS


def get_or_create_dataset(client: rr.catalog.CatalogClient, name: str) -> rr.catalog.DatasetEntry:
    """Catalog dataset entry."""
    try:
        client.create_dataset(name)
    except rr.catalog.AlreadyExistsError:
        pass
    return client.get_dataset(name=name)


def ensure_table(
    client: rr.catalog.CatalogClient,
    *,
    name: str,
    schema: pa.Schema,
    storage_dir: Path,
) -> rr.catalog.TableEntry:
    """
    Create or register a Lance-backed table and register it with DataFusion (client.ctx).
    """
    storage_url = storage_dir.absolute().as_uri()

    try:
        table = client.get_table(name=name)
    except LookupError:
        table = None

    if table is None:
        if storage_dir.exists():
            client.register_table(name=name, url=storage_url)
        else:
            client.create_table(name=name, schema=schema, url=storage_url)
        table = client.get_table(name=name)

    table.reader()
    return table


def ensure_all_tables(client: rr.catalog.CatalogClient) -> None:
    """Ensure all expected tables exist and are registered with client.ctx."""
    ensure_table(
        client,
        name=RUNS_TABLE_NAME,
        schema=RUNS_SCHEMA,
        storage_dir=TABLES_DIR / f"{RUNS_TABLE_NAME}.lance",
    )
    ensure_table(
        client,
        name=EPISODE_METRICS_TABLE_NAME,
        schema=EPISODE_METRIC_SCHEMA,
        storage_dir=TABLES_DIR / f"{EPISODE_METRICS_TABLE_NAME}.lance",
    )


def log_variant_rrd(segment_id: str, layer: str) -> str:
    """Write one .rrd file for (segment_id, layer) and return its file:// URI."""
    rrd_path = RECORDINGS_DIR / f"{segment_id}__{layer}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    # recording_id becomes dataset segment_id when you register the rrd into a dataset
    rec = rr.RecordingStream(
        application_id=f"{EXPERIMENT_FAMILY}/{VERSION_ID}",
        recording_id=segment_id,
    )
    rec.save(str(rrd_path))

    # toy logs (replace with real episode trace)
    rec.log("prompt", rr.TextLog(f"solve segment={segment_id} layer={layer}", level=rr.TextLogLevel.DEBUG))
    rec.log("answer", rr.TextLog(f"hello from {layer} @ {segment_id}"))
    rec.log("debug/value", rr.Scalars([float(np.random.randn())]))

    return rrd_path.absolute().as_uri()


def live_preview_episode(preview: rr.RecordingStream, episode_id: str) -> None:
    """
    ✅ One live recording per run (clean dashboard),
    with episodes separated by entity paths.
    """
    preview.log(f"episodes/{episode_id}/ab/A/answer", rr.TextLog(f"(preview) A answer for {episode_id}"))
    preview.log(f"episodes/{episode_id}/ab/B/answer", rr.TextLog(f"(preview) B answer for {episode_id}"))


def append_run_row(
    runs_table: rr.catalog.TableEntry,
    *,
    experiment_family: str,
    version_id: str,
    run_id: str,
    started_at: float,
    ended_at: float,
) -> None:
    runs_table.append(
        experiment_family=experiment_family,
        version_id=version_id,
        run_id=run_id,
        started_at_unix_s=float(started_at),
        ended_at_unix_s=float(ended_at),
    )


def append_episode_metric_row(
    episode_metrics: rr.catalog.TableEntry,
    *,
    experiment_family: str,
    version_id: str,
    run_id: str,
    suite_name: str,
    task_id: str,
    segment_id: str,
    layer: str,
    rrd_uri: str,
    tokens: int,
    success: bool,
    wall_time_ms: int,
) -> None:
    episode_metrics.append(
        experiment_family=experiment_family,
        version_id=version_id,
        run_id=run_id,
        suite_name=suite_name,
        task_id=task_id,
        segment_id=segment_id,
        layer=layer,
        rrd_uri=rrd_uri,
        tokens=int(tokens),
        success=bool(success),
        wall_time_ms=int(wall_time_ms),
    )


# ------------------- MAIN EXPERIMENT


def run_experiment() -> None:
    started_at = time.time()

    with rr.server.Server() as server:
        client = server.client()

        dataset = get_or_create_dataset(client, EXPERIMENT)

        runs_table = ensure_table(
            client,
            name=RUNS_TABLE_NAME,
            schema=RUNS_SCHEMA,
            storage_dir=TABLES_DIR / f"{RUNS_TABLE_NAME}.lance",
        )

        episode_metrics = ensure_table(
            client,
            name=EPISODE_METRICS_TABLE_NAME,
            schema=EPISODE_METRIC_SCHEMA,
            storage_dir=TABLES_DIR / f"{EPISODE_METRICS_TABLE_NAME}.lance",
        )

        SUITE_NAME = "demo_suite_10"  # ✅ benchmark/split name, not experiment name

        # One live preview recording per run (Tier 0)
        preview = rr.RecordingStream(
            application_id=f"preview/{EXPERIMENT_FAMILY}/{VERSION_ID}",
            recording_id=RUN_ID,  # constant → ONE viewer recording
        )
        preview.spawn()

        for i in range(10):
            task_id = f"ep_{i:06d}"

            # ✅ stable segment identity for cross-run comparisons
            segment_id = f"{SUITE_NAME}/{task_id}"

            live_preview_episode(preview, task_id)

            for layer in ["A", "B"]:  # later: ["tools", "no_tools"] etc
                t0 = time.time()
                rrd_uri = log_variant_rrd(segment_id, layer)

                # Register immediately (Tier 1 canonical)
                dataset.register(rrd_uri, layer_name=layer)

                elapsed_ms = int((time.time() - t0) * 1000)

                tokens = int(np.random.randint(50, 200))
                success = bool(np.random.rand() > 0.3)

                append_episode_metric_row(
                    episode_metrics,
                    experiment_family=EXPERIMENT_FAMILY,
                    version_id=VERSION_ID,
                    run_id=RUN_ID,
                    suite_name=SUITE_NAME,
                    task_id=task_id,
                    segment_id=segment_id,
                    layer=layer,
                    rrd_uri=rrd_uri,
                    tokens=tokens,
                    success=success,
                    wall_time_ms=elapsed_ms,
                )

            time.sleep(0.25)

        ended_at = time.time()
        append_run_row(
            runs_table,
            experiment_family=EXPERIMENT_FAMILY,
            version_id=VERSION_ID,
            run_id=RUN_ID,
            started_at=started_at,
            ended_at=ended_at,
        )

        print("\n--- eval_core sample ---")
        batches = client.ctx.sql(f"SELECT * FROM {RUNS_TABLE_NAME} LIMIT 10").collect()
        for b in batches:
            print(b.to_pandas())

        print("\n--- eval_chomsky sample ---")
        batches = client.ctx.sql(f"SELECT * FROM {EPISODE_METRICS_TABLE_NAME} LIMIT 10").collect()
        for b in batches:
            print(b.to_pandas())

        addr = server.address()
        port = addr.rsplit(":", 1)[-1]

        print(f"\n✅ Dataset server address: {addr}")
        print(f"Open Viewer with:\n  rerun connect 127.0.0.1:{port}\n")
        print(f"Dataset name: {EXPERIMENT}")
        print(f"Run ID: {RUN_ID}")
        print("Press Ctrl+C to stop…")

        while True:
            time.sleep(1)


def run_sql(sql: str) -> None:
    """
    Lightweight SQL runner:
    - starts a tiny local catalog server
    - ensures tables exist
    - runs SQL against client.ctx
    """
    with rr.server.Server() as server:
        client = server.client()
        ensure_all_tables(client)

        batches = client.ctx.sql(sql).collect()
        if not batches:
            print("(0 rows)")
        for b in batches:
            print(b.to_pandas().to_string(index=False))


def sql_repl() -> None:
    """
    Interactive SQL prompt:
      uv run t2f repl

    Commands:
      .tables
      .schema <table>
      .runs
      .quit
    """
    with rr.server.Server() as server:
        client = server.client()
        ensure_all_tables(client)

        print("t2f SQL REPL (DataFusion via Rerun Catalog)")
        print("  .tables                 list tables")
        print("  .schema <table>         describe table")
        print("  .runs                   latest runs")
        print("  .quit                   exit")
        print("")

        while True:
            try:
                q = input("t2f> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return

            if not q:
                continue

            if q in [".quit", ".exit"]:
                return

            if q == ".tables":
                # Try information_schema (DataFusion often provides it)
                try:
                    batches = client.ctx.sql(
                        "SELECT table_name "
                        "FROM information_schema.tables "
                        "WHERE table_schema='public' "
                        "ORDER BY table_name"
                    ).collect()
                    printed = False
                    for b in batches:
                        df = b.to_pandas()
                        if not df.empty:
                            print(df.to_string(index=False))
                            printed = True
                    if not printed:
                        raise RuntimeError("no tables returned")
                except Exception:
                    # Fallback to known table names
                    print("\n".join([RUNS_TABLE_NAME, EPISODE_METRICS_TABLE_NAME]))
                continue

            if q.startswith(".schema"):
                parts = q.split(maxsplit=1)
                if len(parts) != 2:
                    print("usage: .schema <table>")
                    continue
                table = parts[1].strip()
                try:
                    batches = client.ctx.sql(f"DESCRIBE {table}").collect()
                    if not batches:
                        print("(no schema rows)")
                    for b in batches:
                        print(b.to_pandas().to_string(index=False))
                except Exception as e:
                    print(f"schema error: {e}")
                continue

            if q == ".runs":
                try:
                    batches = client.ctx.sql(
                        f"SELECT * FROM {RUNS_TABLE_NAME} "
                        "ORDER BY started_at_unix_s DESC "
                        "LIMIT 20"
                    ).collect()
                    if not batches:
                        print("(0 rows)")
                    for b in batches:
                        print(b.to_pandas().to_string(index=False))
                except Exception as e:
                    print(f"runs error: {e}")
                continue

            try:
                batches = client.ctx.sql(q).collect()
                if not batches:
                    print("(0 rows)")
                    continue
                for b in batches:
                    print(b.to_pandas().to_string(index=False))
            except Exception as e:
                print(f"sql error: {e}")


def main():
    parser = argparse.ArgumentParser(prog="t2f")  # opt in to typer in the future..
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub_sql = sub.add_parser("sql", help="Run a SQL query against the catalog tables")
    sub_sql.add_argument("query", type=str, help='SQL query, e.g. "SELECT * FROM episode_metrics LIMIT 10"')

    sub_repl = sub.add_parser("repl", help="Interactive SQL prompt")

    args = parser.parse_args()

    if args.cmd == "sql":
        run_sql(args.query)
    elif args.cmd == "repl":
        sql_repl()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
