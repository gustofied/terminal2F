from mylogger import setup_logging
import logging
from pathlib import Path

dir = Path(__file__).resolve().parent
logging_config_path = str(dir / "config.json")

setup_logging(logging_config_path)

logger = logging.getLogger(__name__)

logger.debug("A logging example")
