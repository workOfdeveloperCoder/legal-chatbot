from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamDelta:
    """One streamed piece from an LLM: visible answer or hidden thinking."""

    text: str
    kind: str = "token"  # "token" | "thinking"


def coerce_delta(piece: object) -> StreamDelta:
    if isinstance(piece, StreamDelta):
        return piece
    if isinstance(piece, dict):
        return StreamDelta(
            text=str(piece.get("text") or ""),
            kind=str(piece.get("kind") or "token"),
        )
    return StreamDelta(text=str(piece or ""), kind="token")
