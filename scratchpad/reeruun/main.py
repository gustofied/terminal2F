import logging
from pathlib import Path

import rerun as rr

from mylogger import setup_logging


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "scratchpad" / "reeruun" / "config.json"

    rr.init("my_app_logs", spawn=True)  # Rerun must be initialized first
    setup_logging(str(config_path))

    log = logging.getLogger(__name__)
    log.info("info example")
    log.warning("warning example")


if __name__ == "__main__":
    main()
