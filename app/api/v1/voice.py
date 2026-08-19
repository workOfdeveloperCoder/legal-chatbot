"""Local STT/TTS endpoints. Legal answers still go through POST /chat.

STT language is metadata only. The transcript is sent to the existing chat
pipeline, which runs detect_language() on the text itself.
"""
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import get_voice_service
from app.models.user import User
from app.schemas.voice import SpeakRequest, TranscriptResponse, VoiceStatusResponse
from app.services.voice_service import VoiceService

router = APIRouter(
    prefix="/voice",
    tags=["Voice"],
)


@router.get(
    "/status",
    response_model=VoiceStatusResponse,
)
async def voice_status(
    current_user: User = Depends(get_current_active_user),
    service: VoiceService = Depends(get_voice_service),
):
    return service.status()


@router.post(
    "/transcribe",
    response_model=TranscriptResponse,
)
async def transcribe_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    service: VoiceService = Depends(get_voice_service),
):
    return await service.transcribe(file)


@router.post("/speak")
async def speak_text(
    payload: SpeakRequest,
    current_user: User = Depends(get_current_active_user),
    service: VoiceService = Depends(get_voice_service),
):
    audio = service.speak(payload.text)
    return Response(content=audio, media_type="audio/wav")
