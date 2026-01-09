import logging
from pathlib import Path

import rerun as rr

from mylogger import setup_logging


def main() -> None:
    rr.init("my_app_logs", spawn=True)

    config_path = Path(__file__).resolve().parent / "config.json"
    setup_logging(str(config_path))

    log = logging.getLogger(__name__)
    log.info("info example")
    log.warning("warning example")


if __name__ == "__main__":
    main()

