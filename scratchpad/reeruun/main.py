import logging
from mylogger import setup_logging
from pathlib import Path

dir = Path(__name__).resolve().parent

configen = dir / "scratchpad" / "reeruun" / "config.json"

def main() -> None:
    setup_logging(configen)

    log = logging.getLogger(__name__)
    log.debug("debug example")
    log.info("info example")
    log.warning("warning example")


if __name__ == "__main__":
    main()