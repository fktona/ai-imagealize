import logging
import os
from datetime import timedelta

import cv2

from app.core.config import settings
from app.services.detection_service import DetectionService
from app.utils.alert import play_alert_async
from app.utils.run_id import make_run_id

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video analysis workflow."""

    def __init__(self, detection_service: DetectionService) -> None:
        self.detection_service = detection_service

    def analyze(self, video_path: str) -> dict[str, object]:
        run_id = make_run_id()
        run_dir = os.path.join(settings.results_alerts_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_skip = max(1, settings.frame_skip)
        frame_index = 0
        detections = []
        last_alert_time = -settings.sound_alert_cooldown_sec

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % frame_skip != 0:
                    frame_index += 1
                    continue

                result = self.detection_service.detect_frame(frame)
                if result["detections"]:
                    timestamp = self._format_timestamp(frame_index, fps)
                    frame_name = f"alert_{os.path.basename(video_path)}_{frame_index}.jpg"
                    frame_path = os.path.join(run_dir, frame_name)
                    cv2.imwrite(frame_path, result["annotated"])

                    for detection in result["detections"]:
                        detections.append(
                            {
                                **detection,
                                "timestamp": timestamp,
                            }
                        )

                    if any(
                        det["confidence"] >= settings.alert_conf_threshold
                        for det in result["detections"]
                    ):
                        if (frame_index / fps) - last_alert_time >= settings.sound_alert_cooldown_sec:
                            play_alert_async()
                            last_alert_time = frame_index / fps

                    logger.info(
                        "Weapon detected in video",
                        extra={"video_path": video_path, "frame": frame_index, "detections": len(result["detections"])},
                    )

                frame_index += 1
        finally:
            cap.release()

        return {
            "weapon_detected": len(detections) > 0,
            "detections": detections,
            "alert_frames_dir": run_dir,
        }

    @staticmethod
    def _format_timestamp(frame_index: int, fps: float) -> str:
        seconds = frame_index / fps
        td = timedelta(seconds=seconds)
        total_seconds = td.total_seconds()
        return f"{int(total_seconds // 3600):02}:{int((total_seconds % 3600) // 60):02}:{total_seconds % 60:04.1f}"
