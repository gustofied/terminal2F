# src/terminal2f/infra/metrics_store.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import lance

from datafusion import SessionContext


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


@dataclass
class MetricsStore:
    root: Path
    ctx: SessionContext
    table_name: str = "leaderboard"  # stable logical table name

    def leaderboard_path(self) -> Path:
        return self.root / "metrics" / f"{self.table_name}.lance"

    def ensure_dirs(self) -> None:
        self.leaderboard_path().parent.mkdir(parents=True, exist_ok=True)

    def append_row(
        self,
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
        """Append 1 row into the Lance dataset on disk."""
        self.ensure_dirs()

        row = {
            "experiment": experiment,
            "trial_id": trial_id,
            "segment_id": segment_id,
            "variant": variant,
            "tokens": tokens,
            "tool_calls": tool_calls,
            "steps": steps,
            "success": success,
        }

        tbl = pa.Table.from_pylist([row], schema=LEADERBOARD_SCHEMA)

        path = str(self.leaderboard_path())
        mode = "append" if self.leaderboard_path().exists() else "create"
        lance.write_dataset(tbl, path, mode=mode)

    def load_as_arrow(self) -> pa.Table:
        """Read the whole table back as Arrow (fine for MVP / small tables)."""
        ds = lance.dataset(str(self.leaderboard_path()))
        return ds.to_table()

    def register_in_datafusion(self) -> None:
        """
        MVP approach: load Arrow -> register a view.
        For large data you’d want deeper integration, but this is perfect to start.
        DataFusion can register a DataFrame/view for SQL queries. :contentReference[oaicite:4]{index=4}
        """
        arrow_tbl = self.load_as_arrow()
        df = self.ctx.from_arrow(arrow_tbl)  # DataFusion DataFrame from Arrow
        self.ctx.register_view(self.table_name, df)
