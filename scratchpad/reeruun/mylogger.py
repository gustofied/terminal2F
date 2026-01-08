from __future__ import annotations

import atexit
import datetime as dt
import json
import logging
import logging.config
import logging.handlers
from pathlib import Path
from typing import override


LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class UTCISOFormatter(logging.Formatter):
    """%(asctime)s as ISO-8601 with timezone (UTC, 'Z')."""

    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        ts = dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc)
        # ISO-8601 + timezone. Use 'Z' for UTC.
        return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class MyJSONFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys or {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, ensure_ascii=False, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, object]:
        always_fields: dict[str, object] = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        }

        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message: dict[str, object] = {
            out_key: (
                msg_val
                if (msg_val := always_fields.pop(in_key, None)) is not None
                else getattr(record, in_key)
            )
            for out_key, in_key in self.fmt_keys.items()
        }
        message.update(always_fields)

        # Add any extra fields (logger.info(..., extra={...}))
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    """Allow only DEBUG/INFO (<= INFO) through."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= logging.INFO


class AutoStartQueueListener(logging.handlers.QueueListener):
    """QueueListener that starts immediately; respects handler levels by default."""

    def __init__(self, queue, *handlers, respect_handler_level: bool = True):
        super().__init__(queue, *handlers, respect_handler_level=respect_handler_level)
        self.start()


_listener_stop_registered = False


def setup_logging(config_path: str | Path = "config.json", *, queue_handler_name: str = "zzz_queue") -> None:
    """
    Load dictConfig from JSON file, and ensure the queue listener is stopped on exit.
    """
    global _listener_stop_registered

    config_path = Path(config_path)

    # Ensure logs/ exists for the file handler.
    Path("logs").mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    logging.config.dictConfig(config)

    qh = logging.getHandlerByName(queue_handler_name)
    listener = getattr(qh, "listener", None)

    if listener is not None and not _listener_stop_registered:
        atexit.register(listener.stop)
        _listener_stop_registered = True
