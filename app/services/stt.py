from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, status

from app.core.config import settings
from app.services.audio_input import TranscriptResult

logger = logging.getLogger(__name__)


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def whisper_cpp_available() -> bool:
    binary = settings.STT_WHISPER_CPP_BIN or shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    model = settings.STT_WHISPER_CPP_MODEL
    return bool(binary and model and Path(model).is_file())


def resolved_stt_engine() -> str | None:
    engine = (settings.STT_ENGINE or "auto").lower().strip()
    if engine in {"auto", "faster-whisper"} and faster_whisper_available():
        return "faster-whisper"
    if engine in {"auto", "whisper-cpp"} and whisper_cpp_available():
        return "whisper-cpp"
    if engine == "faster-whisper":
        return None
    if engine == "whisper-cpp":
        return None
    if faster_whisper_available():
        return "faster-whisper"
    if whisper_cpp_available():
        return "whisper-cpp"
    return None


def stt_status() -> tuple[bool, str | None, list[str]]:
    engine = resolved_stt_engine()
    missing: list[str] = []
    if engine:
        return True, engine, missing
    missing.append(
        "Install local STT: pip install -r requirements-voice.txt "
        "(faster-whisper). First transcribe downloads the Whisper model "
        f"'{settings.STT_MODEL}' into {settings.STT_MODEL_CACHE}."
    )
    return False, None, missing


@lru_cache
def _whisper_model():
    from faster_whisper import WhisperModel

    cache = Path(settings.STT_MODEL_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading faster-whisper model=%s device=%s compute=%s",
        settings.STT_MODEL,
        settings.STT_DEVICE,
        settings.STT_COMPUTE_TYPE,
    )
    return WhisperModel(
        settings.STT_MODEL,
        device=settings.STT_DEVICE,
        compute_type=settings.STT_COMPUTE_TYPE,
        download_root=str(cache),
    )


def _transcribe_faster_whisper(path: str) -> TranscriptResult:
    model = _whisper_model()
    segments, info = model.transcribe(
        path,
        language=None,
        vad_filter=True,
        beam_size=1,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    language = getattr(info, "language", None)
    duration = getattr(info, "duration", None)
    return TranscriptResult(
        text=text,
        language=str(language) if language else None,
        duration_seconds=float(duration) if duration is not None else None,
        engine="faster-whisper",
    )


def _transcribe_whisper_cpp(path: str) -> TranscriptResult:
    binary = settings.STT_WHISPER_CPP_BIN or shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    model = settings.STT_WHISPER_CPP_MODEL
    if not binary or not model:
        raise _unavailable("whisper.cpp binary or model is not configured.")
    with tempfile.TemporaryDirectory() as tmp:
        output_prefix = os.path.join(tmp, "out")
        command = [
            binary,
            "-m",
            model,
            "-f",
            path,
            "-l",
            "auto",
            "-oj",
            "-of",
            output_prefix,
            "-nt",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("whisper.cpp failed: %s", exc)
            raise _unavailable("Local whisper.cpp transcription failed.") from exc
        payload_path = Path(f"{output_prefix}.json")
        if not payload_path.is_file():
            raise _unavailable("whisper.cpp did not return a transcript.")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    transcription = payload.get("transcription") or payload.get("text") or ""
    if isinstance(transcription, list):
        text = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in transcription
        ).strip()
    else:
        text = str(transcription).strip()
    language = payload.get("result", {}).get("language") if isinstance(payload.get("result"), dict) else payload.get("language")
    return TranscriptResult(
        text=text,
        language=str(language) if language else None,
        duration_seconds=None,
        engine="whisper-cpp",
    )


def transcribe_file(path: str) -> TranscriptResult:
    engine = resolved_stt_engine()
    if engine == "faster-whisper":
        return _transcribe_faster_whisper(path)
    if engine == "whisper-cpp":
        return _transcribe_whisper_cpp(path)
    _, _, missing = stt_status()
    raise _unavailable(missing[0] if missing else "Local speech-to-text is not available.")
