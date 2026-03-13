import logging
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.models.model_loader import get_model

logger = logging.getLogger(__name__)


class DetectionService:
    """Service for running YOLOv8 detections on images and frames."""

    def __init__(self) -> None:
        self.model = get_model()

    def detect_image(self, image_path: str) -> dict[str, Any]:
        results = self.model.predict(source=image_path, conf=settings.conf_threshold, verbose=False)
        result = results[0]

        detections = self._parse_detections(result)
        annotated = result.plot()

        return {
            "detections": detections,
            "annotated": annotated,
        }

    def detect_frame(self, frame: np.ndarray) -> dict[str, Any]:
        results = self.model.predict(source=frame, conf=settings.conf_threshold, verbose=False)
        result = results[0]
        detections = self._parse_detections(result)
        annotated = result.plot()

        return {
            "detections": detections,
            "annotated": annotated,
        }

    def _parse_detections(self, result: Any) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "object": names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections
