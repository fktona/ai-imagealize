import logging
import os

import cv2

from app.core.config import settings
from app.services.detection_service import DetectionService
from app.utils.alert import play_alert_async
from app.utils.run_id import make_run_id

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image analysis workflow."""

    def __init__(self, detection_service: DetectionService) -> None:
        self.detection_service = detection_service

    def analyze(self, image_path: str) -> dict[str, object]:
        """Analyze an image and return detection results."""
        run_id = make_run_id()
        run_dir = os.path.join(settings.results_images_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        result = self.detection_service.detect_image(image_path)
        detections = result["detections"]
        annotated = result["annotated"]

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(run_dir, f"annotated_{base_name}.jpg")
        cv2.imwrite(annotated_path, annotated)

        if any(det["confidence"] >= settings.alert_conf_threshold for det in detections):
            play_alert_async()

        logger.info("Image processed", extra={"image_path": image_path, "detections": len(detections)})

        return {
            "weapon_detected": len(detections) > 0,
            "detections": detections,
            "annotated_image_path": annotated_path,
        }
