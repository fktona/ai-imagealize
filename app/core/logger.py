import logging

from app.core.config import settings


def setup_logging() -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
