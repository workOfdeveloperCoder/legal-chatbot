from __future__ import annotations

import logging
import tempfile

from fastapi import UploadFile

from app.schemas.voice import TranscriptResponse, VoiceStatusResponse
from app.services.audio_input import read_upload
from app.services.stt import stt_status, transcribe_file
from app.services.tts import synthesize_speech, tts_status

logger = logging.getLogger(__name__)


class VoiceService:
    def status(self) -> VoiceStatusResponse:
        stt_ready, stt_engine, stt_missing = stt_status()
        tts_ready, tts_engine, tts_missing = tts_status()
        return VoiceStatusResponse(
            stt_ready=stt_ready,
            tts_ready=tts_ready,
            stt_engine=stt_engine,
            tts_engine=tts_engine,
            missing=[*stt_missing, *tts_missing],
        )

    async def transcribe(self, upload: UploadFile) -> TranscriptResponse:
        data = await read_upload(upload)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
            handle.write(data)
            handle.flush()
            result = transcribe_file(handle.name)
        logger.info(
            "Local STT engine=%s language=%s chars=%s",
            result.engine,
            result.language,
            len(result.text),
        )
        return TranscriptResponse(
            text=result.text,
            stt_language=result.language,
            duration_seconds=result.duration_seconds,
            engine=result.engine,
        )

    def speak(self, text: str) -> bytes:
        audio, engine = synthesize_speech(text)
        logger.info("Local TTS engine=%s bytes=%s", engine, len(audio))
        return audio
