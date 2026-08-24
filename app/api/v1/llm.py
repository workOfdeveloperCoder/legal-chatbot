from fastapi import APIRouter, Depends

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import get_token_usage_service
from app.models.user import User
from app.schemas.llm import (
    LLMStatusResponse,
    TokenCountRequest,
    TokenCountResponse,
)
from app.services.token_usage_service import TokenUsageService

router = APIRouter(
    prefix="/llm",
    tags=["LLM"],
)


@router.get(
    "/status",
    response_model=LLMStatusResponse,
)
async def llm_status(
    current_user: User = Depends(get_current_active_user),
    service: TokenUsageService = Depends(get_token_usage_service),
):
    return service.status()


@router.post(
    "/count-tokens",
    response_model=TokenCountResponse,
)
async def count_tokens(
    payload: TokenCountRequest,
    current_user: User = Depends(get_current_active_user),
    service: TokenUsageService = Depends(get_token_usage_service),
):
    return service.count(
        payload.text,
        extra_texts=payload.extra_texts,
        include_system_prompt=payload.include_system_prompt,
    )
