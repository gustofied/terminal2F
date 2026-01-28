import argparse  # rather typer (later)
import time
from pathlib import Path
import uuid
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.catalog as catalog

# ------------------- DATA SETUP

EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family
VERSION_ID = "v1"  # Specific version of that experiment
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}"  # stable dataset name

LOGS_DIR = Path("logs")
TABLES_DIR = LOGS_DIR / "tables"  # stable across all runs

# Register progress while running (dataset-connect UX)
REGISTER_EVERY_N_TASKS = 2

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
        ("trial_id", pa.string()),  # shared across A/B
        ("episode_id", pa.string()),  # unique per (trial, variant)
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


def dataset_is_empty(dataset: catalog.DatasetEntry) -> bool:
    """
    Best-effort emptiness check for a dataset.
    We treat "any segment rows exist" as non-empty.
    """
    # Prefer 0.28+ API: segment_table()
    try:
        if hasattr(dataset, "segment_table"):
            batches = dataset.segment_table().collect()  
            for b in batches:
                if b.num_rows > 0:
                    return False
            return True
    except Exception:
        pass

    # Fallback: partition_ids() (older APIs)
    try:
        if hasattr(dataset, "segment_ids"):
            return len(dataset.segment_ids()) == 0
        if hasattr(dataset, "partition_ids"):
            return len(dataset.partition_ids()) == 0
    except Exception:
        pass

    # Last resort: manifest() (may be large, but should work)
    try:
        if hasattr(dataset, "manifest"):
            batches = dataset.manifest().collect()
            for b in batches:
                if b.num_rows > 0:
                    return False
            return True
    except Exception:
        pass

    # Conservative default: assume not empty if we cannot prove emptiness
    return False


def reset_dataset(client: catalog.CatalogClient, name: str, *, fallback_name: str) -> tuple[catalog.DatasetEntry, str]:
    """
    Hard-reset a dataset name to be an empty workspace snapshot.
    If we cannot guarantee emptiness, fall back to a unique dataset name.
    """
    delete_ok = True
    try:
        client.get_dataset(name=name).delete()
    except LookupError:
        pass
    except Exception as e:
        delete_ok = False
        print(f"⚠️  Warning: failed to delete dataset '{name}': {e}")

    try:
        client.create_dataset(name)
    except catalog.AlreadyExistsError:
        pass

    ds = client.get_dataset(name=name)

    # Verify emptiness; if not empty, do NOT pretend we're clean.
    if dataset_is_empty(ds):
        return ds, name

    # Fall back to a per-run dataset name (keeps you safe + debuggable).
    print(
        f"⚠️  Warning: dataset '{name}' is not empty after reset "
        f"(delete_ok={delete_ok}). Using fallback dataset '{fallback_name}'."
    )

    actual_fallback = fallback_name
    try:
        client.create_dataset(actual_fallback)
    except catalog.AlreadyExistsError:
        # Ensure uniqueness if somehow already exists
        actual_fallback = f"{fallback_name}__{uuid.uuid4().hex[:8]}"
        client.create_dataset(actual_fallback)

    fallback_ds = client.get_dataset(name=actual_fallback)
    return fallback_ds, actual_fallback


def ensure_table(
    client: catalog.CatalogClient,
    *,
    name: str,
    schema: pa.Schema,
    storage_dir: Path,
) -> catalog.TableEntry:
    """
    Create or register a Lance-backed table and register it with DataFusion (client.ctx).
    """
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
    else:
        # Safety check: ensure required columns exist (avoid silent schema mismatch).
        try:
            existing_schema = table.to_arrow_reader().schema
            existing_cols = {f.name for f in existing_schema}
            expected_cols = {f.name for f in schema}
            missing = sorted(expected_cols - existing_cols)
            if missing:
                raise RuntimeError(
                    f"Table '{name}' is missing required columns {missing}. "
                    f"Expected at least: {sorted(expected_cols)}. Found: {sorted(existing_cols)}. "
                    f"Delete or migrate the existing Lance table at: {storage_dir}"
                )
        except Exception as e:
            # If we can't introspect schema, fail loudly rather than corrupting data.
            raise RuntimeError(f"Failed to validate schema for table '{name}': {e}") from e

    # Ensure DataFusion registration is active.
    table.reader()
    return table


def ensure_all_tables(client: catalog.CatalogClient) -> None:
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
    rrd_path = recordings_dir / problem_id / f"{episode_id}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rec = rr.RecordingStream(
        application_id=f"{EXPERIMENT_FAMILY}/{VERSION_ID}",
        recording_id=problem_id,  # stays stable: dataset segment_id = problem_id
    )
    rec.save(str(rrd_path))

    # Variant separation is handled by dataset layer_name=variant, so don't prefix entity paths with variant.
    rec.log(
        "prompt",
        rr.TextLog(
            f"solve problem={problem_id} episode={episode_id} variant={variant}",
            level=rr.TextLogLevel.DEBUG,
        ),
    )
    rec.log(
        "answer",
        rr.TextLog(f"hello from {variant} @ {problem_id} (episode={episode_id})"),
    )
    rec.log("debug/value", rr.Scalars([float(np.random.randn())]))

    return rrd_path.absolute().as_uri()


def live_preview_episode(preview_a: rr.RecordingStream, preview_b: rr.RecordingStream, problem_id: str) -> None:
    """
    Two live recordings per run (one per variant),
    so variants are separated by recording_id.
    """
    preview_a.log(f"episodes/{problem_id}/answer", rr.TextLog(f"(preview) A answer for {problem_id}"))
    preview_b.log(f"episodes/{problem_id}/answer", rr.TextLog(f"(preview) B answer for {problem_id}"))


def append_run_row(
    runs_table: catalog.TableEntry,
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
    episode_metrics: catalog.TableEntry,
    *,
    experiment_family: str,
    version_id: str,
    run_id: str,
    suite_name: str,
    task_id: str,
    trial_id: str,
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
        trial_id=trial_id,
        episode_id=episode_id,
        problem_id=problem_id,
        variant=variant,
        rrd_uri=rrd_uri,
        tokens=int(tokens),
        success=bool(success),
        wall_time_ms=int(wall_time_ms),
    )


def register_pending(dataset: catalog.DatasetEntry, pending_by_variant: dict[str, list[str]]) -> None:
    """Register whatever we have produced so far into the workspace dataset (batched), then wait."""
    handles = []
    for variant, uris in pending_by_variant.items():
        if uris:
            handles.append(dataset.register(uris, layer_name=variant))
    for h in handles:
        h.wait()


# ------------------- MAIN EXPERIMENT


def run_experiment() -> None:
    run_id = make_run_id()
    recordings_dir = LOGS_DIR / "recordings" / EXPERIMENT_FAMILY / VERSION_ID / run_id
    recordings_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()

    with rr.server.Server() as server:
        client = server.client()

        dataset_name = EXPERIMENT  # single stable dataset name (workspace snapshot)
        dataset, dataset_name = reset_dataset(
            client,
            dataset_name,
            fallback_name=f"{EXPERIMENT}/__broken_reset__/{run_id}",
        )

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
        interrupted = False
        ended_at_eval: float | None = None

        try:
            for i in range(10):
                task_id = f"ep_{i:06d}"
                problem_id = f"{SUITE_NAME}/{task_id}"  # stable across runs
                trial_id = str(uuid.uuid4())  # shared across A/B

                live_preview_episode(preview_a, preview_b, problem_id)

                for variant in ["A", "B"]:
                    episode_id = str(uuid.uuid4())  # unique per (trial, variant)
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
                        trial_id=trial_id,
                        episode_id=episode_id,
                        problem_id=problem_id,
                        variant=variant,
                        rrd_uri=rrd_uri,
                        tokens=tokens,
                        success=success,
                        wall_time_ms=elapsed_ms,
                    )

                # Publish progress to dataset while running (dataset-connect UX)
                if REGISTER_EVERY_N_TASKS > 0 and (i + 1) % REGISTER_EVERY_N_TASKS == 0:
                    try:
                        register_pending(dataset, pending_by_variant)
                        pending_by_variant = {"A": [], "B": []}  # clear only after success
                    except Exception as e:
                        print(f"⚠️  Warning: failed to register progress update: {e}")

                time.sleep(0.25)

        except KeyboardInterrupt:
            interrupted = True

        except Exception as e:
            # Still publish whatever we managed to generate.
            print(f"⚠️  Run crashed mid-loop: {e}")
            interrupted = True

        finally:
            ended_at_eval = time.time()

            # ✅ Always publish remaining progress to the workspace dataset.
            try:
                register_pending(dataset, pending_by_variant)
            except Exception as e:
                print(f"⚠️  Warning: failed to register pending dataset updates: {e}")

            # ✅ Record the run lifecycle (end time = when compute ended, not when you stop viewing).
            append_run_row(
                runs_table,
                experiment_family=EXPERIMENT_FAMILY,
                version_id=VERSION_ID,
                run_id=run_id,
                started_at=started_at,
                ended_at=ended_at_eval,
            )

        # Optional: samples
        try:
            print("\n--- runs sample ---")
            batches = client.ctx.sql(f"SELECT * FROM {RUNS_TABLE_NAME} ORDER BY started_at_unix_s DESC LIMIT 10").collect()
            for b in batches:
                print(b.to_pandas())

            print("\n--- episode_metrics sample ---")
            batches = client.ctx.sql(f"SELECT * FROM {EPISODE_METRICS_TABLE_NAME} ORDER BY run_id DESC LIMIT 10").collect()
            for b in batches:
                print(b.to_pandas())
        except Exception as e:
            print(f"⚠️  Warning: failed to query sample tables: {e}")

        addr = server.address()
        port = addr.rsplit(":", 1)[-1]

        print(f"\n✅ Dataset server address: {addr}")
        print(f"Open Viewer with:\n  rerun connect 127.0.0.1:{port}\n")
        print(f"Dataset name: {dataset_name}")
        print(f"Run ID: {run_id}")
        if interrupted:
            print("⚠️  Run was interrupted; dataset shows partial progress.")
        print("Press Ctrl+C to stop the server…")

        # Keep the server alive for inspection (Ctrl+C now stops the server, not the eval loop).
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


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
