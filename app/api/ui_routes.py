import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.schemas.response_schema import ImageAnalysisResponse, VideoAnalysisResponse
from app.services.detection_service import DetectionService
from app.services.image_processor import ImageProcessor
from app.services.screen_monitor import ScreenMonitor
from app.services.video_processor import VideoProcessor
from app.utils.file_handler import FileValidationError, save_upload_file

router = APIRouter(include_in_schema=False)

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def get_detection_service(request: Request) -> DetectionService:
    return request.app.state.detection_service


def get_screen_monitor(request: Request) -> ScreenMonitor:
    return request.app.state.screen_monitor


def _results_url(file_path: str) -> str:
    abs_path = Path(file_path).resolve()
    results_root = Path("results").resolve()
    return f"/results/{abs_path.relative_to(results_root).as_posix()}"


def _list_runs(root_dir: Path, limit: int = 20) -> list[dict[str, object]]:
    if not root_dir.exists():
        return []
    run_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.name, reverse=True)
    runs: list[dict[str, object]] = []
    for run_dir in run_dirs[:limit]:
        files = [
            _results_url(str(p))
            for p in sorted(run_dir.iterdir())
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        runs.append({"run_id": run_dir.name, "files": files})
    return runs


@router.get("/ui", response_class=HTMLResponse)
async def ui_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "image_result": None,
            "video_result": None,
        },
    )


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@router.get("/streams/wall", response_class=HTMLResponse)
async def streams_wall(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("streams_wall.html", {"request": request})


@router.get("/ui/runs")
async def ui_runs(kind: str) -> dict[str, object]:
    if kind == "images":
        root_dir = Path(settings.results_images_dir)
    elif kind == "videos":
        root_dir = Path(settings.results_alerts_dir)
    elif kind == "screen":
        root_dir = Path(settings.results_screen_alerts_dir)
    else:
        raise HTTPException(status_code=400, detail="Invalid kind")

    return {"runs": _list_runs(root_dir)}


@router.post("/ui/analyze-image", response_class=HTMLResponse)
async def ui_analyze_image(
    request: Request,
    file: UploadFile = File(...),
    detection_service: DetectionService = Depends(get_detection_service),
) -> HTMLResponse:
    try:
        image_path = save_upload_file(file, settings.upload_images_dir, settings.allowed_image_ext)
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    processor = ImageProcessor(detection_service)
    result = processor.analyze(image_path)
    response = ImageAnalysisResponse(**result)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "image_result": {
                "weapon_detected": response.weapon_detected,
                "detections": response.detections,
                "annotated_image_url": _results_url(response.annotated_image_path)
                if response.annotated_image_path
                else None,
            },
            "video_result": None,
        },
    )


@router.post("/ui/analyze-video", response_class=HTMLResponse)
async def ui_analyze_video(
    request: Request,
    file: UploadFile = File(...),
    detection_service: DetectionService = Depends(get_detection_service),
) -> HTMLResponse:
    try:
        video_path = save_upload_file(file, settings.upload_videos_dir, settings.allowed_video_ext)
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    processor = VideoProcessor(detection_service)
    result = processor.analyze(video_path)
    response = VideoAnalysisResponse(**result)

    alert_urls: list[str] = []
    video_name = Path(video_path).name
    if response.alert_frames_dir and os.path.isdir(response.alert_frames_dir):
        for file_name in sorted(os.listdir(response.alert_frames_dir)):
            if video_name in file_name:
                alert_urls.append(_results_url(str(Path(response.alert_frames_dir) / file_name)))
                if len(alert_urls) >= 10:
                    break

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "image_result": None,
            "video_result": {
                "weapon_detected": response.weapon_detected,
                "detections": response.detections,
                "alert_frame_urls": alert_urls,
            },
        },
    )


@router.post("/ui/start-screen-monitor")
async def ui_start_monitor(
    monitor: ScreenMonitor = Depends(get_screen_monitor),
) -> dict[str, str]:
    started = monitor.start()
    return {"status": "started" if started else "already_running"}


@router.post("/ui/stop-screen-monitor")
async def ui_stop_monitor(
    monitor: ScreenMonitor = Depends(get_screen_monitor),
) -> dict[str, str]:
    stopped = monitor.stop()
    return {"status": "stopped" if stopped else "not_running"}
