from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, Query
from fastapi.responses import StreamingResponse
import time
import cv2
import os
import asyncio
import socket
from urllib.parse import urlparse

from app.core.config import settings
from app.schemas.response_schema import ImageAnalysisResponse, VideoAnalysisResponse
from app.services.detection_service import DetectionService
from app.services.image_processor import ImageProcessor
from app.services.screen_monitor import ScreenMonitor
from app.services.camera_monitor import CameraMonitor
from app.services.stream_manager import StreamManager
from app.services.video_processor import VideoProcessor
from app.utils.file_handler import FileValidationError, save_upload_file

router = APIRouter()


def get_detection_service(request: Request) -> DetectionService:
    """Provide a shared DetectionService instance for route handlers."""
    return request.app.state.detection_service


def get_screen_monitor(request: Request) -> ScreenMonitor:
    """Provide the shared ScreenMonitor instance."""
    return request.app.state.screen_monitor


def get_camera_monitor(request: Request) -> CameraMonitor:
    """Provide the shared CameraMonitor instance."""
    return request.app.state.camera_monitor


def get_stream_manager(request: Request) -> StreamManager:
    return request.app.state.stream_manager


def _frame_stream(get_frame_fn):
    def generator():
        interval = 1.0 / max(1, settings.stream_fps)
        try:
            while True:
                frame = get_frame_fn()
                if frame is None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" b"\r\n")
                else:
                    ok, buffer = cv2.imencode(".jpg", frame)
                    if ok:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                        )
                time.sleep(interval)
        except (GeneratorExit, asyncio.CancelledError):
            return

    return generator()


def _validate_rtsp_url(rtsp_url: str) -> None:
    parsed = urlparse(rtsp_url)
    if parsed.scheme not in {"rtsp", "rtsps"}:
        raise HTTPException(status_code=400, detail="Invalid RTSP URL scheme")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="Invalid RTSP URL host")
    port = parsed.port or 554
    try:
        with socket.create_connection((parsed.hostname, port), timeout=2.0):
            pass
    except OSError:
        raise HTTPException(status_code=400, detail="RTSP host not reachable")


@router.post(
    "/analyze-image",
    response_model=ImageAnalysisResponse,
    summary="Analyze an uploaded image",
    description="Upload a single image file and return weapon detections with an annotated image path.",
    responses={
        200: {
            "description": "Detection results",
            "content": {
                "application/json": {
                    "example": {
                        "weapon_detected": True,
                        "detections": [
                            {
                                "object": "gun",
                                "confidence": 0.93,
                                "bbox": [120.5, 80.2, 260.9, 190.6],
                                "timestamp": None,
                            }
                        ],
                        "annotated_image_path": "results/images/2026-03-13_12-10-05/annotated_sample.jpg",
                    }
                }
            },
        },
        400: {"description": "Invalid file extension"},
    },
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze (jpg, png, webp, avif, bmp)."),
    detection_service: DetectionService = Depends(get_detection_service),
) -> ImageAnalysisResponse:
    """Accept an uploaded image, run detection, and return analysis results."""
    try:
        image_path = save_upload_file(file, settings.upload_images_dir, settings.allowed_image_ext)
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    processor = ImageProcessor(detection_service)
    result = processor.analyze(image_path)
    return ImageAnalysisResponse(**result)


@router.post(
    "/analyze-video",
    response_model=VideoAnalysisResponse,
    summary="Analyze an uploaded video",
    description="Upload a video file and return detections with timestamps. Alert frames are saved to disk.",
    responses={
        200: {
            "description": "Detection results",
            "content": {
                "application/json": {
                    "example": {
                        "weapon_detected": True,
                        "detections": [
                            {
                                "object": "knife",
                                "confidence": 0.88,
                                "bbox": [44.0, 99.0, 120.0, 188.0],
                                "timestamp": "00:00:02.1",
                            }
                        ],
                        "alert_frames_dir": "results/alerts/2026-03-13_12-12-40",
                    }
                }
            },
        },
        400: {"description": "Invalid file extension"},
    },
)
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyze (mp4, avi, mov, mkv)."),
    detection_service: DetectionService = Depends(get_detection_service),
) -> VideoAnalysisResponse:
    """Accept an uploaded video, run detection, and return analysis results."""
    try:
        video_path = save_upload_file(file, settings.upload_videos_dir, settings.allowed_video_ext)
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    processor = VideoProcessor(detection_service)
    result = processor.analyze(video_path)
    return VideoAnalysisResponse(**result)


@router.post(
    "/start-screen-monitor",
    include_in_schema=False,
)
async def start_screen_monitor(
    preview: bool | None = Query(
        default=None,
        description="If true, open a local preview window (server machine only).",
        example=True,
    ),
    monitor: ScreenMonitor = Depends(get_screen_monitor),
) -> dict[str, str]:
    """Start background screen monitoring on the server machine."""
    started = monitor.start(preview=preview)
    if not started:
        return {"status": "already_running"}
    return {"status": "started"}


@router.post(
    "/stop-screen-monitor",
    include_in_schema=False,
)
async def stop_screen_monitor(
    monitor: ScreenMonitor = Depends(get_screen_monitor),
) -> dict[str, str]:
    """Stop background screen monitoring on the server machine."""
    stopped = monitor.stop()
    if not stopped:
        return {"status": "not_running"}
    return {"status": "stopped"}


@router.post("/start-webcam-monitor", include_in_schema=False)
async def start_webcam_monitor(
    preview: bool | None = None,
    monitor: CameraMonitor = Depends(get_camera_monitor),
) -> dict[str, str]:
    """Start webcam monitoring on the server machine."""
    started = monitor.start_webcam(preview=preview)
    if not started:
        return {"status": "already_running"}
    return {"status": "started"}


@router.post(
    "/start-rtsp-monitor",
    include_in_schema=False,
    summary="Start a single RTSP monitor",
    description="Starts a single RTSP/IP camera monitor. For multi-feed, use `/streams/start`.",
    responses={
        200: {"description": "Monitor status", "content": {"application/json": {"example": {"status": "started"}}}}
    },
)
async def start_rtsp_monitor(
    url: str = Query(..., description="RTSP URL, e.g. rtsp://user:pass@ip:554/stream", example="rtsp://192.168.1.10:554/stream"),
    preview: bool | None = Query(
        default=None,
        description="If true, open a local preview window (server machine only).",
        example=False,
    ),
    monitor: CameraMonitor = Depends(get_camera_monitor),
) -> dict[str, str]:
    """Start RTSP/IP camera monitoring on the server machine."""
    _validate_rtsp_url(url)
    started = monitor.start_rtsp(url, preview=preview)
    if not started:
        return {"status": "already_running"}
    return {"status": "started"}


@router.post("/stop-camera-monitor", include_in_schema=False)
async def stop_camera_monitor(
    monitor: CameraMonitor = Depends(get_camera_monitor),
) -> dict[str, str]:
    """Stop camera monitoring."""
    stopped = monitor.stop()
    if not stopped:
        return {"status": "not_running"}
    return {"status": "stopped"}


@router.get("/camera/stream", include_in_schema=False)
async def camera_stream(
    monitor: CameraMonitor = Depends(get_camera_monitor),
) -> StreamingResponse:
    return StreamingResponse(
        _frame_stream(monitor.get_latest_frame),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get(
    "/screen/stream",
    include_in_schema=False,
)
async def screen_stream(
    monitor: ScreenMonitor = Depends(get_screen_monitor),
) -> StreamingResponse:
    return StreamingResponse(
        _frame_stream(monitor.get_latest_frame),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post(
    "/streams/start",
    summary="Start RTSP stream worker",
    description="Start a managed RTSP stream by `stream_id` with low-latency capture and detection.",
    responses={
        200: {
            "description": "Stream status",
            "content": {"application/json": {"example": {"status": "started", "stream_id": "drone-01"}}},
        },
        500: {"description": "Max streams reached"},
    },
)
async def start_stream(
    rtsp_url: str = Query(
        ...,
        description="RTSP URL for the drone feed.",
        example="rtsp://192.168.1.20:554/stream",
    ),
    stream_id: str | None = Query(
        default=None,
        description="Optional custom stream id. If omitted, an id is auto-generated.",
        example="drone-01",
    ),
    manager: StreamManager = Depends(get_stream_manager),
    request: Request,
) -> dict[str, str]:
    _validate_rtsp_url(rtsp_url)
    if stream_id is None:
        stream_id = f"stream-{int(time.time())}"
    started = manager.start_stream(stream_id, rtsp_url)
    base_url = str(request.base_url).rstrip("/")
    return {
        "status": "started" if started else "already_running",
        "stream_id": stream_id,
        "preview_url": f"{base_url}/streams/{stream_id}/preview",
        "wall_url": f"{base_url}/streams/wall",
    }


@router.post(
    "/streams/stop",
    summary="Stop RTSP stream worker",
    description="Stop a managed RTSP stream by `stream_id`.",
    responses={
        200: {"description": "Stream status", "content": {"application/json": {"example": {"status": "stopped"}}}}
    },
)
async def stop_stream(
    stream_id: str = Query(..., description="Stream id to stop.", example="drone-01"),
    manager: StreamManager = Depends(get_stream_manager),
) -> dict[str, str]:
    stopped = manager.stop_stream(stream_id)
    return {"status": "stopped" if stopped else "not_running"}


@router.get(
    "/streams/status",
    summary="List stream workers",
    description="List all managed RTSP stream workers and their last detection timestamps.",
    responses={
        200: {
            "description": "Stream list",
            "content": {
                "application/json": {
                    "example": {
                        "streams": [
                            {
                                "stream_id": "drone-01",
                                "rtsp_url": "rtsp://192.168.1.10:554/stream",
                                "running": True,
                                "last_detection_ts": 1710334800.0,
                                "started_at": 1710334700.0,
                            }
                        ]
                    }
                }
            },
        }
    },
)
async def streams_status(
    manager: StreamManager = Depends(get_stream_manager),
) -> dict[str, object]:
    return {"streams": [s.__dict__ for s in manager.list_status()]}


@router.get(
    "/streams/recordings",
    summary="List stream recordings",
    description="List available recorded video files for all streams.",
    responses={
        200: {
            "description": "Recording list",
            "content": {
                "application/json": {
                    "example": {
                        "recordings": [
                            {
                                "stream_id": "drone-01",
                                "run_id": "drone-01_2026-03-13_12-20-10",
                                "url": "/recordings/drone-01_2026-03-13_12-20-10/stream_record.avi",
                            }
                        ]
                    }
                }
            },
        }
    },
)
async def streams_recordings(request: Request) -> dict[str, object]:
    root = settings.results_stream_recordings_dir
    recordings = []
    base_url = str(request.base_url).rstrip("/")
    if os.path.isdir(root):
        for run_dir in sorted(os.listdir(root), reverse=True):
            run_path = os.path.join(root, run_dir)
            if not os.path.isdir(run_path):
                continue
            for file_name in os.listdir(run_path):
                if file_name.endswith((".avi", ".mp4")):
                    url_path = f"/recordings/{run_dir}/{file_name}"
                    recordings.append(
                        {
                            "stream_id": run_dir.split("_")[0],
                            "run_id": run_dir,
                            "url": f"{base_url}{url_path}",
                        }
                    )
    return {"recordings": recordings}


@router.get(
    "/streams/{stream_id}/preview",
    summary="Stream preview (MJPEG)",
    description="MJPEG preview of the latest annotated frame for a given stream.",
    responses={
        200: {"description": "MJPEG stream"},
        404: {"description": "Stream not found"},
    },
)
async def stream_preview(
    stream_id: str,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamingResponse:
    worker = manager.get_worker(stream_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    return StreamingResponse(
        _frame_stream(worker.get_latest_frame),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
