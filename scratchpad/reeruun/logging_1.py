import logging

logger = logging.getLogger("my_app")

handler = logging.FileHandler('my_module.log')
logger.addHandler(handler)

def main():
    logging.basicConfig(level="DEBUG")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning messae")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("exception message")
    
if __name__ == "__main__":
    main()