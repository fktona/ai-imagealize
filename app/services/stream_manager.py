import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2

from app.core.config import settings
from app.services.detection_service import DetectionService
from app.utils.alert import play_alert_async
from app.utils.run_id import make_run_id

logger = logging.getLogger(__name__)


@dataclass
class StreamStatus:
    stream_id: str
    rtsp_url: str
    running: bool
    last_detection_ts: float | None
    started_at: float | None


class StreamWorker:
    def __init__(self, stream_id: str, rtsp_url: str, detection_service: DetectionService) -> None:
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.detection_service = detection_service
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_annotated = None
        self._running = False
        self._run_dir = None
        self._alert_count = 0
        self._last_alert_time = -settings.stream_alert_cooldown_sec
        self._persist_counter = 0
        self._last_detection_ts: float | None = None
        self._started_at: float | None = None
        self._record_path: str | None = None

    def start(self) -> None:
        run_id = make_run_id()
        self._run_dir = os.path.join(settings.results_camera_alerts_dir, f"{self.stream_id}_{run_id}")
        os.makedirs(self._run_dir, exist_ok=True)
        if settings.enable_stream_recording:
            record_dir = os.path.join(settings.results_stream_recordings_dir, f"{self.stream_id}_{run_id}")
            os.makedirs(record_dir, exist_ok=True)
            self._record_path = os.path.join(record_dir, f"stream_record{settings.stream_record_ext}")
        self._stop_event.clear()
        self._started_at = time.time()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()
        self._running = True

    def stop(self) -> None:
        self._stop_event.set()
        self._running = False

    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()

    def get_latest_frame(self):
        with self._frame_lock:
            return None if self._latest_annotated is None else self._latest_annotated.copy()

    def status(self) -> StreamStatus:
        return StreamStatus(
            stream_id=self.stream_id,
            rtsp_url=self.rtsp_url,
            running=self.is_running(),
            last_detection_ts=self._last_detection_ts,
            started_at=self._started_at,
        )

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.rtsp_url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            logger.error("Unable to open RTSP stream", extra={"stream_id": self.stream_id})
            self._stop_event.set()
            return

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                with self._frame_lock:
                    self._latest_frame = frame
                time.sleep(0.001)
        finally:
            cap.release()

    def _inference_loop(self) -> None:
        interval = 1.0 / max(1, settings.stream_inference_fps)
        record_interval = 1.0 / max(1, settings.stream_record_fps)
        last_record_time = 0.0
        video_writer: cv2.VideoWriter | None = None
        writer_ready = False
        record_path = self._record_path
        while not self._stop_event.is_set():
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            result = self.detection_service.detect_frame(frame)
            annotated = result["annotated"]
            detections = result["detections"]

            with self._frame_lock:
                self._latest_annotated = annotated

            if detections:
                self._persist_counter += 1
                if self._persist_counter >= settings.alert_persist_frames:
                    now = time.time()
                    if now - self._last_alert_time >= settings.stream_alert_cooldown_sec:
                        if any(det["confidence"] >= settings.alert_conf_threshold for det in detections):
                            play_alert_async()
                            self._last_alert_time = now
                            self._last_detection_ts = now
                            if self._run_dir:
                                filename = f"alert_{self.stream_id}_{int(now)}.jpg"
                                cv2.imwrite(os.path.join(self._run_dir, filename), annotated)
            else:
                self._persist_counter = 0

            if settings.enable_stream_recording and record_path:
                if not writer_ready:
                    height, width = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*settings.stream_record_fourcc)
                    video_writer = cv2.VideoWriter(
                        record_path,
                        fourcc,
                        float(settings.stream_record_fps),
                        (width, height),
                    )
                    if video_writer.isOpened():
                        writer_ready = True
                        logger.info("Stream recording started", extra={"stream_id": self.stream_id, "path": record_path})
                    else:
                        logger.warning("Stream recording failed to start", extra={"stream_id": self.stream_id})
                        video_writer.release()
                        video_writer = None
                        record_path = None
                if video_writer is not None and (time.time() - last_record_time) >= record_interval:
                    try:
                        video_writer.write(annotated)
                        last_record_time = time.time()
                    except Exception as exc:
                        logger.warning("Stream recording stopped: %s", exc)
                        video_writer.release()
                        video_writer = None

            time.sleep(interval)

        if video_writer is not None:
            video_writer.release()


class StreamManager:
    def __init__(self, detection_service: DetectionService) -> None:
        self.detection_service = detection_service
        self._streams: Dict[str, StreamWorker] = {}
        self._lock = threading.Lock()

    def start_stream(self, stream_id: str, rtsp_url: str) -> bool:
        with self._lock:
            if stream_id in self._streams and self._streams[stream_id].is_running():
                return False
            if len(self._streams) >= settings.max_streams:
                raise RuntimeError("Max streams reached")
            worker = StreamWorker(stream_id, rtsp_url, self.detection_service)
            self._streams[stream_id] = worker
            worker.start()
            return True

    def stop_stream(self, stream_id: str) -> bool:
        with self._lock:
            worker = self._streams.get(stream_id)
            if not worker:
                return False
            worker.stop()
            return True

    def list_status(self) -> list[StreamStatus]:
        with self._lock:
            return [worker.status() for worker in self._streams.values()]

    def get_worker(self, stream_id: str) -> Optional[StreamWorker]:
        with self._lock:
            return self._streams.get(stream_id)
