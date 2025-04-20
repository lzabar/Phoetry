import logging


def get_logger(name=__name__):
    """
    Function to initialize the logger
    Should be called in any other .py by :

    logger = get_logger(name=__name__)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(
            filename='log.log',
            encoding='utf-8',
            level=logging.DEBUG,
            format="[%(asctime)s] [%(levelname)s] [%(name)s/%(funcName)s]--> %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )

        return logger
