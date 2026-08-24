from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
import json

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import get_chat_service
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    service: ChatService = Depends(get_chat_service),
):

    return await service.chat(
        user=current_user,
        payload=payload,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    service: ChatService = Depends(get_chat_service),
):
    async def event_publisher():
        stream = service.chat_stream(
            user=current_user,
            payload=payload,
        )
        try:
            async for item in stream:
                if await request.is_disconnected():
                    break
                event = item.get("event", "message")
                yield _sse(event, item)
        except Exception:
            yield _sse(
                "error",
                {
                    "event": "error",
                    "status": 503,
                    "detail": "The assistant is temporarily unavailable.",
                },
            )
        finally:
            await stream.aclose()

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )