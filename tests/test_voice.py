from __future__ import annotations

import io
import wave

import pytest
from fastapi import HTTPException

from app.rag.language import UserLanguage, detect_language
from app.services.audio_input import validate_audio_bytes, wav_duration_seconds
from app.services.tts import pick_espeak_voice


def _silent_wav(seconds: float = 0.2, rate: int = 16000) -> bytes:
    frames = b"\x00\x00" * int(rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


def test_validate_audio_rejects_empty():
    with pytest.raises(HTTPException) as exc:
        validate_audio_bytes(b"")
    assert exc.value.status_code == 400


def test_validate_audio_rejects_non_wav():
    with pytest.raises(HTTPException) as exc:
        validate_audio_bytes(b"not-a-wav", filename="clip.mp3")
    assert exc.value.status_code == 400


def test_validate_audio_accepts_wav():
    validate_audio_bytes(_silent_wav(), filename="clip.wav")
    duration = wav_duration_seconds(_silent_wav())
    assert duration is not None
    assert 0.1 < duration < 0.4


def test_tts_voice_uses_existing_language_detector():
    urdu = detect_language("دفعہ 302 کے تحت سزا کیا ہے؟")
    assert urdu.response_language == UserLanguage.UR_PK
    assert pick_espeak_voice(urdu.response_language) == "ur"

    punjabi = detect_language("eh case di bail application kithay file karni ae?")
    assert punjabi.response_language == UserLanguage.PA_PK
    assert pick_espeak_voice(punjabi.response_language) == "pa"

    english = detect_language("Explain bail under section 497 PPC.")
    assert english.response_language == UserLanguage.EN_PK
    assert pick_espeak_voice(english.response_language) == "en"


def test_voice_stack_does_not_use_openai(monkeypatch):
    from app.core import config

    assert not hasattr(config.settings, "OPENAI_REALTIME_MODEL")
    assert getattr(config.settings, "STT_ENGINE", None) == "auto"
    assert getattr(config.settings, "TTS_ENGINE", None) == "auto"
