#!/usr/bin/env python3
"""Install/download local Voice Mode assets. Does not call any paid API."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = ROOT / "models" / "voice"
PIPER_EN_ONNX = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
PIPER_EN_JSON = PIPER_EN_ONNX + ".json"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up local STT/TTS for Legal GPT Voice Mode")
    parser.add_argument("--download-piper-en", action="store_true")
    args = parser.parse_args()

    print(f"Python {sys.version}")
    print(f"espeak-ng: {shutil.which('espeak-ng') or shutil.which('espeak') or 'not found'}")
    try:
        import faster_whisper  # noqa: F401

        print("faster-whisper: installed")
    except ImportError:
        print("faster-whisper: missing — pip install -r requirements-voice.txt")
    try:
        import piper  # noqa: F401

        print("piper-tts: installed")
    except ImportError:
        print("piper-tts: missing — pip install -r requirements-voice.txt")

    if args.download_piper_en:
        onnx = VOICE_DIR / "en_US-lessac-medium.onnx"
        cfg = VOICE_DIR / "en_US-lessac-medium.onnx.json"
        download(PIPER_EN_ONNX, onnx)
        download(PIPER_EN_JSON, cfg)
        print(f"Piper English voice saved to {onnx}")

    print("\nWhisper models download automatically on first /voice/transcribe")
    print("into models/voice/whisper (STT_MODEL=base by default).")
    print("Urdu/Punjabi TTS uses espeak-ng unless TTS_VOICE_UR / TTS_VOICE_PA are set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
