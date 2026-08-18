from __future__ import annotations

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]

    temperature: float = 0.1

    max_tokens: int = 2048


class ChatCompletionResponse(BaseModel):
    content: str

    prompt_tokens: int | None = None

    completion_tokens: int | None = None

    total_tokens: int | None = None
    