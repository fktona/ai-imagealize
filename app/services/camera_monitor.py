import logging
import os
import threading
import time

import cv2

from app.core.config import settings
from app.services.detection_service import DetectionService
from app.utils.alert import play_alert_async
from app.utils.run_id import make_run_id

logger = logging.getLogger(__name__)


class CameraMonitor:
    """Monitor a webcam or RTSP stream for weapons and optionally preview annotated frames."""

    def __init__(self, detection_service: DetectionService) -> None:
        self.detection_service = detection_service
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._source: str | int | None = None
        self._run_dir: str | None = None
        self._preview_enabled: bool | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame: cv2.Mat | None = None
        self._record_path: str | None = None

    def start_webcam(self, device_index: int = 0, preview: bool | None = None) -> bool:
        return self._start(device_index, preview)

    def start_rtsp(self, url: str, preview: bool | None = None) -> bool:
        return self._start(url, preview)

    def stop(self) -> bool:
        with self._lock:
            if not self._running:
                return False
            self._stop_event.set()
            self._running = False
            return True

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_latest_frame(self) -> cv2.Mat | None:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _start(self, source: str | int, preview: bool | None) -> bool:
        with self._lock:
            if self._running:
                return False
            self._stop_event.clear()
            self._source = source
            self._preview_enabled = preview
            run_id = make_run_id()
            self._run_dir = os.path.join(settings.results_camera_alerts_dir, run_id)
            os.makedirs(self._run_dir, exist_ok=True)
            if settings.enable_camera_recording:
                record_dir = os.path.join(settings.results_camera_recordings_dir, run_id)
                os.makedirs(record_dir, exist_ok=True)
                self._record_path = os.path.join(record_dir, f"camera_record{settings.camera_record_ext}")
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._running = True
            return True

    def _run(self) -> None:
        source = self._source if self._source is not None else 0
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Unable to open camera source", extra={"source": source})
            self._stop_event.set()
            return

        last_alert_time = -settings.sound_alert_cooldown_sec
        frame_skip = max(1, settings.frame_skip)
        frame_index = 0
        last_save_time = 0.0
        preview_enabled = self._preview_enabled if self._preview_enabled is not None else settings.enable_screen_preview
        window_name = "Camera Monitor (Weapon Detection)"
        last_annotated = None
        video_writer: cv2.VideoWriter | None = None
        writer_ready = False
        record_path = self._record_path

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                annotated = frame
                if frame_index % frame_skip == 0:
                    result = self.detection_service.detect_frame(frame)
                    annotated = result["annotated"]
                    last_annotated = annotated
                    if result["detections"]:
                        now = time.time()
                        if now - last_save_time >= 0.5:
                            filename = f"camera_alert_{int(now)}_{frame_index}.jpg"
                            run_dir = self._run_dir or settings.results_camera_alerts_dir
                            path = os.path.join(run_dir, filename)
                            cv2.imwrite(path, annotated)
                            last_save_time = now

                        if any(
                            det["confidence"] >= settings.alert_conf_threshold
                            for det in result["detections"]
                        ):
                            if now - last_alert_time >= settings.sound_alert_cooldown_sec:
                                play_alert_async()
                                last_alert_time = now

                        logger.info(
                            "Weapon detected on camera",
                            extra={"source": source, "detections": len(result["detections"])},
                        )
                elif last_annotated is not None:
                    annotated = last_annotated

                if settings.enable_camera_recording and record_path and not writer_ready:
                    height, width = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*settings.camera_record_fourcc)
                    video_writer = cv2.VideoWriter(
                        record_path,
                        fourcc,
                        float(settings.camera_record_fps),
                        (width, height),
                    )
                    if video_writer.isOpened():
                        writer_ready = True
                        logger.info("Camera recording started", extra={"path": record_path})
                    else:
                        logger.warning("Camera recording failed to start", extra={"path": record_path})
                        video_writer.release()
                        video_writer = None
                        record_path = None

                if video_writer is not None:
                    try:
                        video_writer.write(annotated)
                    except Exception as exc:
                        logger.warning("Camera recording stopped: %s", exc)
                        video_writer.release()
                        video_writer = None

                with self._frame_lock:
                    self._latest_frame = annotated

                if preview_enabled:
                    try:
                        cv2.imshow(window_name, annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self._stop_event.set()
                    except Exception as exc:
                        preview_enabled = False
                        logger.warning("Camera preview disabled: %s", exc)

                frame_index += 1
                time.sleep(0.01)
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if preview_enabled:
                cv2.destroyWindow(window_name)
