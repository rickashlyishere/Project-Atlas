from core.logging.logger import setup_logger

logger = setup_logger(__name__)

logger.info("Project Atlas started successfully.")
logger.warning("Logger is working.")
logger.error("This is a test error.")