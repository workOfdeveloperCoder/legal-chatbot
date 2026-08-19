from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptResponse(BaseModel):
    text: str
    stt_language: str | None = None
    duration_seconds: float | None = None
    engine: str | None = None


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class VoiceStatusResponse(BaseModel):
    stt_ready: bool
    tts_ready: bool
    stt_engine: str | None = None
    tts_engine: str | None = None
    missing: list[str] = Field(default_factory=list)
