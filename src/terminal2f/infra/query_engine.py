# src/terminal2f/infra/query_engine.py
from __future__ import annotations
from datafusion import SessionContext


def make_ctx() -> SessionContext:
    ctx = SessionContext()

    # Keep it simple: one catalog (default) + schemas for domains.
    # If schema exists, this is safe.
    ctx.sql("CREATE SCHEMA IF NOT EXISTS metrics").collect()

    # (Optional) set defaults so unqualified table names resolve under `metrics`
    # ctx.set_current_schema("metrics")

    return ctx
