from pydantic import BaseModel, Field


class Detection(BaseModel):
    object: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[float] | None = None
    timestamp: str | None = None
    frame_url: str | None = None


class ImageAnalysisResponse(BaseModel):
    weapon_detected: bool
    detections: list[Detection]
    annotated_image_path: str | None = None
    annotated_image_url: str | None = None


class VideoAnalysisResponse(BaseModel):
    weapon_detected: bool
    detections: list[Detection]
    alert_frames_dir: str | None = None
