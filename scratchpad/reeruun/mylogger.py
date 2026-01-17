from __future__ import annotations

import datetime as dt
import json
import logging
import logging.config
from pathlib import Path
from typing import override


LOG_RECORD_BUILTIN_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class UTCISOFormatter(logging.Formatter):
    @override
    def formatTime(self, record: logging.LogRecord, datefmt=None) -> str:
        ts = dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc)
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class MyJSONFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys or {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        always: dict[str, object] = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        }
        if record.exc_info:
            always["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            always["stack_info"] = self.formatStack(record.stack_info)

        out = {
            k: (always.pop(v, None) if v in always else getattr(record, v))
            for k, v in self.fmt_keys.items()
        }
        out.update(always)

        for k, v in record.__dict__.items():
            if k not in LOG_RECORD_BUILTIN_ATTRS:
                out[k] = v

        return json.dumps(out, ensure_ascii=False, default=str)


class NonErrorFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= logging.INFO


def setup_logging(config_path: str = "logging/config.json") -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    logging.config.dictConfig(cfg)
