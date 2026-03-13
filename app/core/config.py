from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "weapon-detection-api"
    model_path: str = "models/best.pt"

    upload_images_dir: str = "uploads/images"
    upload_videos_dir: str = "uploads/videos"

    results_images_dir: str = "results/images"
    results_alerts_dir: str = "results/alerts"
    results_screen_alerts_dir: str = "results/screen/alerts"

    frame_skip: int = 5
    conf_threshold: float = 0.25
    alert_conf_threshold: float = 0.3
    enable_sound_alert: bool = False
    sound_alert_cooldown_sec: float = 2.0
    enable_screen_preview: bool = False
    enable_screen_recording: bool = True
    screen_record_fps: int = 12
    screen_record_fourcc: str = "MJPG"
    screen_record_ext: str = ".avi"
    stream_fps: int = 15
    max_streams: int = 10
    stream_preview_fps: int = 15
    stream_inference_fps: int = 10
    alert_persist_frames: int = 3
    stream_alert_cooldown_sec: float = 2.0
    results_stream_recordings_dir: str = "results/streams/recordings"
    enable_stream_recording: bool = True
    stream_record_fps: int = 10
    stream_record_fourcc: str = "MJPG"
    stream_record_ext: str = ".avi"
    results_screen_recordings_dir: str = "results/screen/recordings"
    results_camera_alerts_dir: str = "results/camera/alerts"
    results_camera_recordings_dir: str = "results/camera/recordings"
    enable_camera_recording: bool = True
    camera_record_fps: int = 24
    camera_record_fourcc: str = "MJPG"
    camera_record_ext: str = ".avi"

    allowed_image_ext: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif")
    allowed_video_ext: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "WEAPON_DETECT_"
        protected_namespaces = ("settings_",)


settings = Settings()
