from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.components import TextLogLevel
from datetime import datetime, timezone

# ------------------- DATA SETUP

EXPERIMENT_FAMILY = "TOOLS_VS_NOTOOLS"  # Experiment family 
VERSION_ID = "v1"      # Specific version of that experiment
EXPERIMENT = f"{EXPERIMENT_FAMILY}/{VERSION_ID}" # (this is *The* Experiment you are doing) The experiment is dervied from famliy and version
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

RUNS_TABLE_NAME = "runs"
EVAL_CORE_TABLE_NAME = "eval_core"
EVAL_CHOMSKY_TABLE_NAME = "eval_chomsky"


RUNS_SCHEMA = pa.schema(
    [
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("mode", pa.string()),              # "eval" | "train"
        ("suite_name", pa.string()),        # optional (eval suites)
        ("started_at_unix_s", pa.float64()),
        ("ended_at_unix_s", pa.float64()),
        ("status", pa.string()),            # "running" | "done" | "failed"
        ("stop_reason", pa.string()),       # "max_episodes" | "timeout" | ...
        ("git_commit", pa.string()),        # optional
        ("config_json", pa.string()),       # optional
    ]
)

EVAL_CORE_SCHEMA = pa.schema(
    [
        # join keys (universal)
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("segment_id", pa.string()),
        ("layer", pa.string()),
        # minimal cross-experiment metrics
        ("tokens", pa.int64()),
        ("tool_calls", pa.int64()),
        ("steps", pa.int64()),
        ("success", pa.bool_()),
        ("wall_time_ms", pa.int64()),
        ("score", pa.float64()),  # optional numeric score; can be null
    ]
)

EVAL_CHOMSKY_SCHEMA = pa.schema(
    [
        # join keys (same as eval_core)
        ("experiment_family", pa.string()),
        ("version_id", pa.string()),
        ("run_id", pa.string()),
        ("segment_id", pa.string()),
        ("layer", pa.string()),
        # automata / memory architecture
        ("agent_class", pa.string()),           # "FA" | "PDA" | "TM"
        ("memory_mode", pa.string()),           # "none" | "stack" | "scratchpad"
        ("context_window_tokens", pa.int64()),  # bounded buffer size (if relevant)
        ("stack_depth_max", pa.int64()),        # if PDA-style
        ("scratchpad_bytes_written", pa.int64()),  # if TM-style
        ("scratchpad_reads", pa.int64()),
        ("scratchpad_writes", pa.int64()),
        # safety/verification-oriented signals
        ("termination_kind", pa.string()),      # "halted" | "timeout" | "crashed"
        ("loop_detected", pa.bool_()),
        ("safety_violation_flag", pa.bool_()),
        ("unsafe_action_count", pa.int64()),
        ("risk_score", pa.float64()),           # optional; null if unused
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

    table.reader()  # make queryable via client.ctx.sql(...)
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
        name=EVAL_CORE_TABLE_NAME,
        schema=EVAL_CORE_SCHEMA,
        storage_dir=TABLES_DIR / f"{EVAL_CORE_TABLE_NAME}.lance",
    )
    ensure_table(
        client,
        name=EVAL_CHOMSKY_TABLE_NAME,
        schema=EVAL_CHOMSKY_SCHEMA,
        storage_dir=TABLES_DIR / f"{EVAL_CHOMSKY_TABLE_NAME}.lance",
    )


def log_variant_rrd(segment_id: str, layer: str) -> str:
    """Write one .rrd file for (segment_id, layer) and return its file:// URI."""
    rrd_path = RECORDINGS_DIR / f"{segment_id}__{layer}.rrd"

    # recording_id becomes dataset segment_id when you register the rrd into a dataset
    rec = rr.RecordingStream(
        application_id=f"{EXPERIMENT_FAMILY}/{VERSION_ID}",
        recording_id=segment_id,
    )
    rec.save(str(rrd_path))

    # toy logs (replace with real episode trace)
    rec.log("prompt", rr.TextLog(f"solve segment={segment_id} layer={layer}", level=TextLogLevel.DEBUG))
    rec.log("answer", rr.TextLog(f"hello from {layer} @ {segment_id}"))
    rec.log("debug/value", rr.Scalars([float(np.random.randn())]))

    return rrd_path.absolute().as_uri()


def live_preview_episode(
    episode_id: str,
    viewer_started: bool,
) -> bool:
    """
    ✅ One live recording per run (clean dashboard),
    with episodes separated by entity paths.
    """
    preview = rr.RecordingStream(
        application_id=f"preview/{EXPERIMENT_FAMILY}/{VERSION_ID}",
        recording_id=RUN_ID,  # constant → ONE viewer recording
    )

    if not viewer_started:
        preview.spawn()
        viewer_started = True
    else:
        preview.connect_grpc()

    preview.log(f"episodes/{episode_id}/ab/A/answer", rr.TextLog(f"(preview) A answer for {episode_id}"))
    preview.log(f"episodes/{episode_id}/ab/B/answer", rr.TextLog(f"(preview) B answer for {episode_id}"))

    return viewer_started


def append_run_row(
    runs_table: rr.catalog.TableEntry,
    *,
    experiment_family: str,
    version_id: str,
    run_id: str,
    mode: str,
    suite_name: Optional[str],
    started_at: float,
    ended_at: float,
    status: str,
    stop_reason: str,
    git_commit: Optional[str] = None,
    config_json: Optional[str] = None,
) -> None:
    runs_table.append(
        experiment_family=experiment_family,
        version_id=version_id,
        run_id=run_id,
        mode=mode,
        suite_name=suite_name,
        started_at_unix_s=float(started_at),
        ended_at_unix_s=float(ended_at),
        status=status,
        stop_reason=stop_reason,
        git_commit=git_commit,
        config_json=config_json,
    )


def append_eval_core_row(
    eval_core: rr.catalog.TableEntry,
    *,
    experiment_family: str,
    version_id: str,
    run_id: str,
    segment_id: str,
    layer: str,
    tokens: int,
    tool_calls: int,
    steps: int,
    success: bool,
    wall_time_ms: int,
    score: Optional[float] = None,
) -> None:
    eval_core.append(
        experiment_family=experiment_family,
        version_id=version_id,
        run_id=run_id,
        segment_id=segment_id,
        layer=layer,
        tokens=int(tokens),
        tool_calls=int(tool_calls),
        steps=int(steps),
        success=bool(success),
        wall_time_ms=int(wall_time_ms),
        score=None if score is None else float(score),
    )


def append_eval_chomsky_row(
    eval_chomsky: rr.catalog.TableEntry,
    *,
    experiment_family: str,
    version_id: str,
    run_id: str,
    segment_id: str,
    layer: str,
    agent_class: str,
    memory_mode: str,
    context_window_tokens: Optional[int],
    stack_depth_max: Optional[int],
    scratchpad_bytes_written: Optional[int],
    scratchpad_reads: Optional[int],
    scratchpad_writes: Optional[int],
    termination_kind: str,
    loop_detected: bool,
    safety_violation_flag: bool,
    unsafe_action_count: int,
    risk_score: Optional[float] = None,
) -> None:
    eval_chomsky.append(
        experiment_family=experiment_family,
        version_id=version_id,
        run_id=run_id,
        segment_id=segment_id,
        layer=layer,
        agent_class=agent_class,
        memory_mode=memory_mode,
        context_window_tokens=context_window_tokens,
        stack_depth_max=stack_depth_max,
        scratchpad_bytes_written=scratchpad_bytes_written,
        scratchpad_reads=scratchpad_reads,
        scratchpad_writes=scratchpad_writes,
        termination_kind=termination_kind,
        loop_detected=bool(loop_detected),
        safety_violation_flag=bool(safety_violation_flag),
        unsafe_action_count=int(unsafe_action_count),
        risk_score=None if risk_score is None else float(risk_score),
    )


# ------------------- MAIN EXPERIMENT

def run_experiment() -> None:
    viewer_started = False
    started_at = time.time()

    # Treat this script as an eval sweep for demo purposes
    MODE = "eval"
    SUITE_NAME = "demo_suite_10"
    STOP_REASON = "max_episodes"

    # For the Chomsky/automata study (set these from your actual agent config)
    AGENT_CLASS = "TM"        # "FA" | "PDA" | "TM"
    MEMORY_MODE = "scratchpad"  # "none" | "stack" | "scratchpad"
    CONTEXT_WINDOW_TOKENS = 8192

    with rr.server.Server() as server:
        client = server.client()

        # ✅ Stable dataset per experiment/version
        dataset = get_or_create_dataset(client, EXPERIMENT)

        # ✅ Ensure tables
        runs_table = ensure_table(
            client,
            name=RUNS_TABLE_NAME,
            schema=RUNS_SCHEMA,
            storage_dir=TABLES_DIR / f"{RUNS_TABLE_NAME}.lance",
        )
        eval_core = ensure_table(
            client,
            name=EVAL_CORE_TABLE_NAME,
            schema=EVAL_CORE_SCHEMA,
            storage_dir=TABLES_DIR / f"{EVAL_CORE_TABLE_NAME}.lance",
        )
        eval_chomsky = ensure_table(
            client,
            name=EVAL_CHOMSKY_TABLE_NAME,
            schema=EVAL_CHOMSKY_SCHEMA,
            storage_dir=TABLES_DIR / f"{EVAL_CHOMSKY_TABLE_NAME}.lance",
        )

        rrd_uris: list[str] = []
        rrd_layers: list[str] = []

        # “episode/task identity” loop (10 datapoints)
        for i in range(10):
            episode_id = f"ep_{i:06d}"

            # ✅ Segment ID must be globally unique within the dataset
            # because dataset persists across runs.
            segment_id = f"run={RUN_ID}__{episode_id}"

            viewer_started = live_preview_episode(episode_id, viewer_started)

            for layer in ["A", "B"]:
                t0 = time.time()
                rrd_uri = log_variant_rrd(segment_id, layer)
                elapsed_ms = int((time.time() - t0) * 1000)

                rrd_uris.append(rrd_uri)
                rrd_layers.append(layer)

                tokens = int(np.random.randint(50, 200))
                tool_calls = int(np.random.randint(0, 5))
                steps = int(np.random.randint(1, 12))
                success = bool(np.random.rand() > 0.3)

                # ✅ Universal metrics (always)
                append_eval_core_row(
                    eval_core,
                    experiment_family=EXPERIMENT_FAMILY,
                    version_id=VERSION_ID,
                    run_id=RUN_ID,
                    segment_id=segment_id,
                    layer=layer,
                    tokens=tokens,
                    tool_calls=tool_calls,
                    steps=steps,
                    success=success,
                    wall_time_ms=elapsed_ms,
                    score=None,
                )

                # ✅ Chomsky/automata instrumentation (your research focus)
                append_eval_chomsky_row(
                    eval_chomsky,
                    experiment_family=EXPERIMENT_FAMILY,
                    version_id=VERSION_ID,
                    run_id=RUN_ID,
                    segment_id=segment_id,
                    layer=layer,
                    agent_class=AGENT_CLASS,
                    memory_mode=MEMORY_MODE,
                    context_window_tokens=CONTEXT_WINDOW_TOKENS,
                    stack_depth_max=None,                 # fill when PDA
                    scratchpad_bytes_written=1234,        # example
                    scratchpad_reads=5,
                    scratchpad_writes=3,
                    termination_kind="halted",
                    loop_detected=False,
                    safety_violation_flag=False,
                    unsafe_action_count=0,
                    risk_score=None,
                )

            time.sleep(0.05)

        # ✅ Register recordings into the dataset as A/B layers
        handle = dataset.register(rrd_uris, layer_name=rrd_layers)
        handle.wait()

        ended_at = time.time()
        append_run_row(
            runs_table,
            experiment_family=EXPERIMENT_FAMILY,
            version_id=VERSION_ID,
            run_id=RUN_ID,
            mode=MODE,
            suite_name=SUITE_NAME,
            started_at=started_at,
            ended_at=ended_at,
            status="done",
            stop_reason=STOP_REASON,
            git_commit=None,
            config_json=None,
        )

        # Example query: show rows we just wrote
        print("\n--- eval_core sample ---")
        batches = client.ctx.sql(f"SELECT * FROM {EVAL_CORE_TABLE_NAME} LIMIT 10").collect()
        for b in batches:
            print(b.to_pandas())

        print("\n--- eval_chomsky sample ---")
        batches = client.ctx.sql(f"SELECT * FROM {EVAL_CHOMSKY_TABLE_NAME} LIMIT 10").collect()
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
                    print("\n".join([RUNS_TABLE_NAME, EVAL_CORE_TABLE_NAME, EVAL_CHOMSKY_TABLE_NAME]))
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

            # Normal SQL
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
    parser = argparse.ArgumentParser(prog="t2f")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub_sql = sub.add_parser("sql", help="Run a SQL query against the catalog tables")
    sub_sql.add_argument("query", type=str, help='SQL query, e.g. "SELECT * FROM eval_core LIMIT 10"')

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
