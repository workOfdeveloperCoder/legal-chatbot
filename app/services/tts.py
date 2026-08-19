from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, status

from app.core.config import settings
from app.rag.language import UserLanguage, detect_language

logger = logging.getLogger(__name__)


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def _espeak_bin() -> str | None:
    configured = settings.TTS_ESPEAK_BIN
    return shutil.which(configured) or shutil.which("espeak-ng") or shutil.which("espeak")


def piper_voice_path(language: UserLanguage) -> Path | None:
    mapping = {
        UserLanguage.EN_PK: settings.TTS_VOICE_EN,
        UserLanguage.UR_PK: settings.TTS_VOICE_UR or settings.TTS_VOICE_EN,
        UserLanguage.PA_PK: settings.TTS_VOICE_PA or settings.TTS_VOICE_UR or settings.TTS_VOICE_EN,
        UserLanguage.MIXED: settings.TTS_VOICE_EN,
    }
    raw = mapping.get(language) or settings.TTS_VOICE_EN
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def piper_available() -> bool:
    if piper_voice_path(UserLanguage.EN_PK):
        try:
            import piper  # noqa: F401

            return True
        except ImportError:
            try:
                import piper.voice  # noqa: F401

                return True
            except ImportError:
                binary = settings.TTS_PIPER_BIN or shutil.which("piper")
                return bool(binary)
    return False


def espeak_available() -> bool:
    return _espeak_bin() is not None


def resolved_tts_engine() -> str | None:
    engine = (settings.TTS_ENGINE or "auto").lower().strip()
    if engine in {"auto", "piper"} and piper_available():
        return "piper"
    if engine in {"auto", "espeak-ng", "espeak"} and espeak_available():
        return "espeak-ng"
    if piper_available():
        return "piper"
    if espeak_available():
        return "espeak-ng"
    return None


def tts_status() -> tuple[bool, str | None, list[str]]:
    engine = resolved_tts_engine()
    if engine:
        return True, engine, []
    return False, None, [
        "Install local TTS: pip install -r requirements-voice.txt and run "
        "`python scripts/setup_voice.py --download-piper-en`, or install espeak-ng."
    ]


def pick_espeak_voice(language: UserLanguage) -> str:
    if language == UserLanguage.UR_PK:
        return "ur"
    if language == UserLanguage.PA_PK:
        return "pa"
    return "en"


@lru_cache
def _load_piper_voice(path: str):
    try:
        from piper import PiperVoice
    except ImportError:
        from piper.voice import PiperVoice

    return PiperVoice.load(path)


def _synthesize_piper(text: str, language: UserLanguage) -> bytes:
    voice_path = piper_voice_path(language)
    if voice_path is None:
        raise _unavailable("Piper voice model file is missing.")
    try:
        import wave

        from piper import PiperVoice  # noqa: F401
    except ImportError:
        binary = settings.TTS_PIPER_BIN or shutil.which("piper")
        if not binary:
            raise _unavailable("piper-tts is not installed.")
        return _synthesize_piper_cli(text, voice_path, binary)

    voice = _load_piper_voice(str(voice_path))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        _piper_synthesize(voice, text, wav_file)
    return buffer.getvalue()


def _piper_synthesize(voice, text: str, wav_file) -> None:
    if hasattr(voice, "synthesize_wav"):
        voice.synthesize_wav(text, wav_file)
        return
    voice.synthesize(text, wav_file)


def _synthesize_piper_cli(text: str, voice_path: Path, binary: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
        try:
            subprocess.run(
                [binary, "--model", str(voice_path), "--output_file", handle.name],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("piper CLI failed: %s", exc)
            raise _unavailable("Piper could not synthesize speech.") from exc
        return Path(handle.name).read_bytes()


def _synthesize_espeak(text: str, language: UserLanguage) -> bytes:
    binary = _espeak_bin()
    if not binary:
        raise _unavailable("espeak-ng is not installed.")
    voice = pick_espeak_voice(language)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
        try:
            subprocess.run(
                [
                    binary,
                    "-v",
                    voice,
                    "-s",
                    "165",
                    "-w",
                    handle.name,
                    text,
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("espeak-ng failed: %s", exc)
            raise _unavailable("espeak-ng could not synthesize speech.") from exc
        return Path(handle.name).read_bytes()


def synthesize_speech(text: str) -> tuple[bytes, str]:
    spoken = text.strip()
    if not spoken:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nothing to speak.",
        )
    detection = detect_language(spoken)
    language = detection.response_language
    engine = resolved_tts_engine()

    # Prefer eSpeak for Urdu/Punjabi unless a dedicated Piper voice exists.
    dedicated = {
        UserLanguage.UR_PK: settings.TTS_VOICE_UR,
        UserLanguage.PA_PK: settings.TTS_VOICE_PA,
    }
    needs_native = language in {UserLanguage.UR_PK, UserLanguage.PA_PK}
    has_native_piper = bool(dedicated.get(language) and Path(str(dedicated.get(language))).is_file())

    if needs_native and espeak_available() and not has_native_piper:
        return _synthesize_espeak(spoken, language), "espeak-ng"
    if engine == "piper":
        return _synthesize_piper(spoken, language), "piper"
    if engine == "espeak-ng":
        return _synthesize_espeak(spoken, language), "espeak-ng"
    _, _, missing = tts_status()
    raise _unavailable(missing[0] if missing else "Local text-to-speech is not available.")
