from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.ui_routes import router as ui_router
from app.core.config import settings
from app.core.logger import setup_logging
from app.models.model_loader import get_model
from app.services.detection_service import DetectionService
from app.services.camera_monitor import CameraMonitor
from app.services.stream_manager import StreamManager
from app.services.screen_monitor import ScreenMonitor


def create_app() -> FastAPI:
    setup_logging()
    application = FastAPI(title=settings.app_name)
    application.include_router(router)
    application.include_router(ui_router)
    application.mount("/results", StaticFiles(directory="results"), name="results")
    application.mount(
        "/recordings",
        StaticFiles(directory=settings.results_stream_recordings_dir),
        name="stream-recordings",
    )

    @application.on_event("startup")
    def startup_event() -> None:
        # Ensure directories exist
        for path in [
            settings.upload_images_dir,
            settings.upload_videos_dir,
            settings.results_images_dir,
            settings.results_alerts_dir,
            settings.results_screen_alerts_dir,
            settings.results_screen_recordings_dir,
            settings.results_camera_alerts_dir,
            settings.results_camera_recordings_dir,
            settings.results_stream_recordings_dir,
        ]:
            import os

            os.makedirs(path, exist_ok=True)

        # Warm model
        get_model()
        # Shared services
        detection_service = DetectionService()
        application.state.detection_service = detection_service
        application.state.screen_monitor = ScreenMonitor(detection_service)
        application.state.camera_monitor = CameraMonitor(detection_service)
        application.state.stream_manager = StreamManager(detection_service)

    return application


app = create_app()
