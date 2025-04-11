import sys
from pathlib import Path
from loguru import logger
from .config import settings

# Remove default handler
logger.remove()

# Add console handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
)

# Add file handler
logger.add(
    settings.LOG_FILE,
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level=settings.LOG_LEVEL,
)

# Export logger
__all__ = ["logger"] 