import logging
import threading
import time

import cv2
import mss
import numpy as np
import os

from app.core.config import settings
from app.services.detection_service import DetectionService
from app.utils.alert import play_alert_async

logger = logging.getLogger(__name__)


class ScreenMonitor:
    """Continuously capture the entire screen and run weapon detection."""

    def __init__(self, detection_service: DetectionService) -> None:
        self.detection_service = detection_service
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._run_dir: str | None = None
        self._record_path: str | None = None
        self._preview_enabled: bool | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame: cv2.Mat | None = None

    def start(self, preview: bool | None = None) -> bool:
        with self._lock:
            if self._running:
                return False
            self._stop_event.clear()
            self._preview_enabled = preview
            run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
            self._run_dir = f"{settings.results_screen_alerts_dir}/{run_id}"
            os.makedirs(self._run_dir, exist_ok=True)
            if settings.enable_screen_recording:
                record_dir = f"{settings.results_screen_recordings_dir}/{run_id}"
                os.makedirs(record_dir, exist_ok=True)
                self._record_path = f"{record_dir}/screen_record{settings.screen_record_ext}"
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            self._running = True
            return True

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

    def _run(self) -> None:
        last_alert_time = -settings.sound_alert_cooldown_sec
        frame_skip = max(1, settings.frame_skip)
        frame_index = 0
        last_save_time = 0.0
        window_name = "Screen Monitor (Weapon Detection)"
        preview_enabled = self._preview_enabled if self._preview_enabled is not None else settings.enable_screen_preview
        video_writer: cv2.VideoWriter | None = None
        last_annotated = None
        writer_ready = False
        record_path = self._record_path
        target_interval = 1.0 / max(1, settings.screen_record_fps)
        last_tick = time.time()

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor
                # Initialize writer lazily after first frame to ensure correct size.
                while not self._stop_event.is_set():
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    annotated = frame
                    if frame_index % frame_skip == 0:
                        result = self.detection_service.detect_frame(frame)
                        annotated = result["annotated"]
                        last_annotated = annotated
                        if result["detections"]:
                            now = time.time()
                            if now - last_save_time >= 0.5:
                                filename = f"screen_alert_{int(now)}_{frame_index}.jpg"
                                run_dir = self._run_dir or settings.results_screen_alerts_dir
                                path = f"{run_dir}/{filename}"
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
                                "Weapon detected on screen",
                                extra={"detections": len(result["detections"])},
                            )
                    elif last_annotated is not None:
                        annotated = last_annotated

                    with self._frame_lock:
                        self._latest_frame = annotated

                    if settings.enable_screen_recording and record_path and not writer_ready:
                        height, width = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*settings.screen_record_fourcc)
                        video_writer = cv2.VideoWriter(
                            record_path,
                            fourcc,
                            float(settings.screen_record_fps),
                            (width, height),
                        )
                        if video_writer.isOpened():
                            writer_ready = True
                            logger.info("Screen recording started", extra={"path": record_path})
                        else:
                            logger.warning("Screen recording failed to start", extra={"path": record_path})
                            video_writer.release()
                            video_writer = None
                            record_path = None

                    if video_writer is not None:
                        try:
                            video_writer.write(annotated)
                        except Exception as exc:
                            logger.warning("Screen recording stopped: %s", exc)
                            video_writer.release()
                            video_writer = None

                    if preview_enabled:
                        try:
                            cv2.imshow(window_name, annotated)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                self._stop_event.set()
                        except Exception as exc:
                            preview_enabled = False
                            logger.warning("Screen preview disabled: %s", exc)

                    frame_index += 1
                    elapsed = time.time() - last_tick
                    sleep_for = max(0.0, target_interval - elapsed)
                    if sleep_for:
                        time.sleep(sleep_for)
                    last_tick = time.time()
        finally:
            if video_writer is not None:
                video_writer.release()
            if preview_enabled:
                cv2.destroyWindow(window_name)
