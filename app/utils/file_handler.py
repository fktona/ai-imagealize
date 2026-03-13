import os
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""


def _validate_extension(filename: str, allowed_ext: tuple[str, ...]) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed_ext:
        raise FileValidationError(f"Unsupported file extension: {ext}")


def save_upload_file(upload_file: UploadFile, target_dir: str, allowed_ext: tuple[str, ...]) -> str:
    """Save an UploadFile to disk and return its path."""

    _validate_extension(upload_file.filename, allowed_ext)

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    ext = Path(upload_file.filename).suffix.lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(target_dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path
