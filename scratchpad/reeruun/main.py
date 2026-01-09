import logging
from pathlib import Path

import rerun as rr

from mylogger import setup_logging


def main() -> None:
    rr.init("my_app_logs", spawn=True)
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config.json"

    listener = setup_logging(str(config_path))
    log = logging.getLogger(__name__)

    try:
        log.info("info example")
        log.warning("warning example")
    finally:
        if listener:
            listener.stop()  # drains queue + joins listener thread       
        rr.disconnect()  # closes connections/files :contentReference[oaicite:3]{index=3}


if __name__ == "__main__":
    main()
