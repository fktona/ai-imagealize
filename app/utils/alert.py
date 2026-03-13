import logging
import threading

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


def _generate_beep(sample_rate: int = 44100, duration_sec: float = 0.25, frequency: int = 880) -> bytes:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = (tone * 32767).astype(np.int16)
    return audio.tobytes()


def _play_sound() -> None:
    try:
        import simpleaudio as sa  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Sound alert disabled (simpleaudio not available): %s", exc)
        return

    try:
        audio_data = _generate_beep()
        sa.play_buffer(audio_data, num_channels=1, bytes_per_sample=2, sample_rate=44100)
    except Exception as exc:  # pragma: no cover
        logger.warning("Sound alert failed: %s", exc)


def play_alert_async() -> None:
    if not settings.enable_sound_alert:
        return
    thread = threading.Thread(target=_play_sound, daemon=True)
    thread.start()
