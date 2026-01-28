import argparse  # rather typer (later)
import time
from pathlib import Path
import uuid
from datetime import datetime, timezone
import rerun.catalog as catalog
import numpy as np
import pyarrow as pa
import rerun as rr

# ------------------- DATA SETUP

EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"  # stable dataset name

LOGS_DIR = Path("logs")
TABLES_DIR = LOGS_DIR / "tables"  # stable across all runs

# ------------------- DATA MODEL

RUNS_TABLE_NAME = "runs"
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
        ("task_id", pa.string()),  # stable task key within suite
        ("episode_id", pa.string()),  # unique attempt id (UUID) per (problem, run)
        ("problem_id", pa.string()),  # stable problem id (suite/task)
        ("variant", pa.string()),  # A/B
        ("rrd_uri", pa.string()),  # artifact pointer
        ("tokens", pa.int64()),
        ("success", pa.bool_()),
        ("wall_time_ms", pa.int64()),
    ]
)

# ------------------- HELPERS


def make_run_id() -> str:
    # millisecond precision UTC timestamp
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]


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
    # Avoid import-time side effects: only create dirs when we actually need them.
    storage_dir.parent.mkdir(parents=True, exist_ok=True)

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

    # Ensure DataFusion registration is active.
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


def log_variant_rrd(recordings_dir: Path, problem_id: str, episode_id: str, variant: str) -> str:
    """Write one .rrd file for (problem_id, episode_id, variant) and return its file:// URI."""
    rrd_path = recordings_dir / problem_id / f"{episode_id}__{variant}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rec = rr.RecordingStream(
        application_id=f"{EXPERIMENT_FAMILY}/{VERSION_ID}",
        recording_id=problem_id,  # stays stable for Option S dataset segment
    )
    rec.save(str(rrd_path))

    rec.log(
        f"{variant}/prompt",
        rr.TextLog(
            f"solve problem={problem_id} episode={episode_id} variant={variant}",
            level=rr.TextLogLevel.DEBUG,
        ),
    )
    rec.log(
        f"{variant}/answer",
        rr.TextLog(f"hello from {variant} @ {problem_id} (episode={episode_id})"),
    )
    rec.log(f"{variant}/debug/value", rr.Scalars([float(np.random.randn())]))

    return rrd_path.absolute().as_uri()


def live_preview_episode(preview_a: rr.RecordingStream, preview_b: rr.RecordingStream, problem_id: str) -> None:
    """
    Two live recordings per run (one per variant),
    so variants are separated by recording_id.
    """
    preview_a.log(f"episodes/{problem_id}/answer", rr.TextLog(f"(preview) A answer for {problem_id}"))
    preview_b.log(f"episodes/{problem_id}/answer", rr.TextLog(f"(preview) B answer for {problem_id}"))


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
    episode_id: str,
    problem_id: str,
    variant: str,
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
        episode_id=episode_id,
        problem_id=problem_id,
        variant=variant,
        rrd_uri=rrd_uri,
        tokens=int(tokens),
        success=bool(success),
        wall_time_ms=int(wall_time_ms),
    )


# ------------------- MAIN EXPERIMENT


def run_experiment() -> None:
    # ✅ run-specific stuff belongs here (not at import time)
    run_id = make_run_id()
    recordings_dir = LOGS_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / run_id
    recordings_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()

    with rr.server.Server() as server:
        client = server.client()

        dataset_name = EXPERIMENT  # stable dataset (latest snapshot index)
        dataset = get_or_create_dataset(client, dataset_name)

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

        SUITE_NAME = "demo_suite_10"  # benchmark/split name

        # Two live preview recordings per run (Tier 0), separated by recording_id
        preview_a = rr.RecordingStream(
            application_id=f"preview/{EXPERIMENT_FAMILY}/{VERSION_ID}",
            recording_id=f"{run_id}/A",
        )
        preview_a.spawn()

        preview_b = rr.RecordingStream(
            application_id=f"preview/{EXPERIMENT_FAMILY}/{VERSION_ID}",
            recording_id=f"{run_id}/B",
        )
        preview_b.spawn()

        pending_by_variant: dict[str, list[str]] = {"A": [], "B": []}

        try:
            for i in range(10):
                task_id = f"ep_{i:06d}"
                problem_id = f"{SUITE_NAME}/{task_id}"  # stable across runs
                episode_id = str(uuid.uuid4())

                live_preview_episode(preview_a, preview_b, problem_id)

                for variant in ["A", "B"]:
                    t0 = time.time()
                    rrd_uri = log_variant_rrd(recordings_dir, problem_id, episode_id, variant)
                    pending_by_variant[variant].append(rrd_uri)

                    elapsed_ms = int((time.time() - t0) * 1000)

                    tokens = int(np.random.randint(50, 200))
                    success = bool(np.random.rand() > 0.3)

                    append_episode_metric_row(
                        episode_metrics,
                        experiment_family=EXPERIMENT_FAMILY,
                        version_id=VERSION_ID,
                        run_id=run_id,
                        suite_name=SUITE_NAME,
                        task_id=task_id,
                        episode_id=episode_id,
                        problem_id=problem_id,
                        variant=variant,
                        rrd_uri=rrd_uri,
                        tokens=tokens,
                        success=success,
                        wall_time_ms=elapsed_ms,
                    )

                time.sleep(0.25)

            # Register batched (Tier 1 canonical) and wait once per variant
            handles = []
            for variant, uris in pending_by_variant.items():
                if uris:
                    handles.append(dataset.register(uris, layer_name=variant))
            for h in handles:
                h.wait()

            print("\n--- runs sample ---")
            batches = client.ctx.sql(f"SELECT * FROM {RUNS_TABLE_NAME} LIMIT 10").collect()
            for b in batches:
                print(b.to_pandas())

            print("\n--- episode_metrics sample ---")
            batches = client.ctx.sql(f"SELECT * FROM {EPISODE_METRICS_TABLE_NAME} LIMIT 10").collect()
            for b in batches:
                print(b.to_pandas())

            addr = server.address()
            port = addr.rsplit(":", 1)[-1]

            print(f"\n✅ Dataset server address: {addr}")
            print(f"Open Viewer with:\n  rerun connect 127.0.0.1:{port}\n")
            print(f"Dataset name: {dataset_name}")
            print(f"Run ID: {run_id}")
            print("Press Ctrl+C to stop…")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            pass

        finally:
            ended_at = time.time()
            append_run_row(
                runs_table,
                experiment_family=EXPERIMENT_FAMILY,
                version_id=VERSION_ID,
                run_id=run_id,
                started_at=started_at,
                ended_at=ended_at,
            )


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="t2f")
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
