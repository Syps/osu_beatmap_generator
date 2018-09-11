import logging

from osu_map_gen.util.definitions import LOG_FILE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(LOG_FILE)
handler.setLevel(logging.INFO)

# create a logging format
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# add the handlers to the logger
logger.addHandler(handler)
logger.info('logger ready')
