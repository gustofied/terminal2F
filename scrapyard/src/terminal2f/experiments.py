from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from typing import Any, Optional

from .telemetry_rerun import SegmentRecorder


@dataclass
class ExperimentController:
    """
    dataset_name (experiment)
      └── segments/  (trials)
            ├── <segment_name>_<recording_id>.rrd
            └── manifest.jsonl
    """

    dataset_name: str
    out_root: Path = Path("recordings")
    application_id: str = "terminal2f"
    spawn_viewer: bool = False

    def __post_init__(self) -> None:
        self.dataset_dir = self.out_root / self.dataset_name
        self.segments_dir = self.dataset_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.dataset_dir / "manifest.jsonl"

    def new_segment(
        self,
        segment_name: str,
        *,
        meta: Optional[dict[str, Any]] = None,
    ) -> SegmentRecorder:
        rid = uuid4().hex
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in segment_name.strip())
        rrd_path = self.segments_dir / f"{safe_name}_{rid}.rrd"

        entry = {
            "dataset": self.dataset_name,
            "segment": safe_name,
            "recording_id": rid,
            "rrd_path": str(rrd_path),
            "ts": time.time(),
            "meta": meta or {},
        }

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return SegmentRecorder(
            dataset=self.dataset_name,
            segment=safe_name,
            recording_id=rid,
            rrd_path=rrd_path,
            application_id=self.application_id,
            spawn_viewer=self.spawn_viewer,
            meta=meta or {},
        )
