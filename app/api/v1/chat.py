import asyncio
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import get_chat_service
from app.models.user import User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    QuickActionResponse,
    QuickActionsListResponse,
)
from app.services.chat_service import ChatService
from app.services.quick_actions import list_quick_actions
from app.core.config import settings
from app.search.web_search import WebSearchService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

@router.get(
    "/quick-actions",
    response_model=QuickActionsListResponse,
    response_model_by_alias=True,
)
async def quick_actions(
    current_user: User = Depends(get_current_active_user),
):
    del current_user
    searcher = WebSearchService()
    return QuickActionsListResponse(
        quick_actions=[
            QuickActionResponse(
                id=action.id,
                slug=action.slug,
                title=action.title,
                description=action.description,
                icon=action.icon,
                prompt=action.prompt,
                requires_document=action.requires_document,
                enables_web_search=action.enables_web_search,
            )
            for action in list_quick_actions()
        ],
        web_search_enabled=settings.WEB_SEARCH_ENABLED,
        web_search_provider=(
            searcher.resolved_provider() if settings.WEB_SEARCH_ENABLED else None
        ),
    )


@router.post(
    "",
    response_model=ChatResponse,
    response_model_by_alias=True,
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
    current_user: User = Depends(get_current_active_user),
    service: ChatService = Depends(get_chat_service),
):
    async def event_publisher():
        stream = service.chat_stream(
            user=current_user,
            payload=payload,
        )
        iterator = stream.__aiter__()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(iterator.__anext__(), timeout=10)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    # Real SSE events (not comments) so Vite/Safari proxies
                    # treat the connection as alive during long R1 thinking.
                    yield _sse(
                        "status",
                        {
                            "event": "status",
                            "phase": "thinking",
                            "detail": "Still generating — local model is working…",
                        },
                    )
                    continue
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