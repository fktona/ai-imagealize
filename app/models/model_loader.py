from functools import lru_cache

from ultralytics import YOLO

from app.core.config import settings


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    """Load and cache the YOLOv8 model."""

    return YOLO(settings.model_path)
