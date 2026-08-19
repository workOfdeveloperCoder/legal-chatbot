from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "application/octet-stream",
}


@dataclass(slots=True, frozen=True)
class TranscriptResult:
    text: str
    language: str | None
    duration_seconds: float | None
    engine: str


def validate_audio_bytes(data: bytes, *, filename: str | None = None) -> None:
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio was empty.",
        )
    if len(data) > settings.STT_MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Audio is too large. Keep recordings under 45 seconds.",
        )
    name = (filename or "").lower()
    if name and not name.endswith((".wav", ".wave")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload a WAV recording.",
        )
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload a WAV recording.",
        )


def wav_duration_seconds(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return None
            duration = frames / float(rate)
    except wave.Error:
        return None
    if duration > settings.STT_MAX_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio is longer than {settings.STT_MAX_SECONDS} seconds.",
        )
    return duration


async def read_upload(upload: UploadFile) -> bytes:
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload a WAV recording.",
        )
    data = await upload.read()
    validate_audio_bytes(data, filename=upload.filename)
    wav_duration_seconds(data)
    return data
