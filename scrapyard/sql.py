
# def run_sql(sql: str) -> None:
#     """
#     Lightweight SQL runner:
#     - starts a tiny local catalog server
#     - ensures tables exist
#     - runs SQL against client.ctx
#     """
#     with rr.server.Server() as server:
#         client = server.client()
#         ensure_all_tables(client)

#         batches = client.ctx.sql(sql).collect()
#         if not batches:
#             print("(0 rows)")
#         for b in batches:
#             print(b.to_pandas().to_string(index=False))


# def sql_repl() -> None:
#     """
#     Interactive SQL prompt:
#       uv run t2f repl

#     Commands:
#       .tables
#       .schema <table>
#       .runs
#       .quit
#     """
#     with rr.server.Server() as server:
#         client = server.client()
#         ensure_all_tables(client)

#         print("t2f SQL REPL (DataFusion via Rerun Catalog)")
#         print("  .tables                 list tables")
#         print("  .schema <table>         describe table")
#         print("  .runs                   latest runs")
#         print("  .quit                   exit")
#         print("")

#         while True:
#             try:
#                 q = input("t2f> ").strip()
#             except (EOFError, KeyboardInterrupt):
#                 print("\nbye")
#                 return

#             if not q:
#                 continue

#             if q in [".quit", ".exit"]:
#                 return

#             if q == ".tables":
#                 try:
#                     batches = client.ctx.sql(
#                         "SELECT table_name "
#                         "FROM information_schema.tables "
#                         "WHERE table_schema='public' "
#                         "ORDER BY table_name"
#                     ).collect()
#                     printed = False
#                     for b in batches:
#                         df = b.to_pandas()
#                         if not df.empty:
#                             print(df.to_string(index=False))
#                             printed = True
#                     if not printed:
#                         raise RuntimeError("no tables returned")
#                 except Exception:
#                     print("\n".join([RUNS_TABLE_NAME, EPISODE_METRICS_TABLE_NAME]))
#                 continue

#             if q.startswith(".schema"):
#                 parts = q.split(maxsplit=1)
#                 if len(parts) != 2:
#                     print("usage: .schema <table>")
#                     continue
#                 table = parts[1].strip()
#                 try:
#                     batches = client.ctx.sql(f"DESCRIBE {table}").collect()
#                     if not batches:
#                         print("(no schema rows)")
#                     for b in batches:
#                         print(b.to_pandas().to_string(index=False))
#                 except Exception as e:
#                     print(f"schema error: {e}")
#                 continue

#             if q == ".runs":
#                 try:
#                     batches = client.ctx.sql(
#                         f"SELECT * FROM {RUNS_TABLE_NAME} "
#                         "ORDER BY started_at_unix_s DESC "
#                         "LIMIT 20"
#                     ).collect()
#                     if not batches:
#                         print("(0 rows)")
#                     for b in batches:
#                         print(b.to_pandas().to_string(index=False))
#                 except Exception as e:
#                     print(f"runs error: {e}")
#                 continue

#             try:
#                 batches = client.ctx.sql(q).collect()
#                 if not batches:
#                     print("(0 rows)")
#                     continue
#                 for b in batches:
#                     print(b.to_pandas().to_string(index=False))
#             except Exception as e:
#                 print(f"sql error: {e}")


# def main() -> None:
#     parser = argparse.ArgumentParser(prog="t2f")
#     sub = parser.add_subparsers(dest="cmd", required=False)

#     sub_sql = sub.add_parser("sql", help="Run a SQL query against the catalog tables")
#     sub_sql.add_argument("query", type=str, help='SQL query, e.g. "SELECT * FROM episode_metrics LIMIT 10"')

#     sub_repl = sub.add_parser("repl", help="Interactive SQL prompt")

#     args = parser.parse_args()

#     if args.cmd == "sql":
#         run_sql(args.query)
#     elif args.cmd == "repl":
#         sql_repl()
#     else:
#         run_experiment()