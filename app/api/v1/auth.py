from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Response, status

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import get_auth_service
from app.auth.service import AuthService
from app.models.user import User
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    RegisterRequest,
    RegisterResponse,
    UserResponse,
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    payload: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
):
    user = await service.register(payload)

    return RegisterResponse(
        user=UserResponse.model_validate(user),
    )


@router.post(
    "/login",
    response_model=LoginResponse,
)
async def login(
    payload: LoginRequest,
    service: AuthService = Depends(get_auth_service),
):
    return await service.login(payload)


@router.post(
    "/refresh",
    response_model=RefreshTokenResponse,
)
async def refresh(
    payload: RefreshTokenRequest,
    service: AuthService = Depends(get_auth_service),
):
    return await service.refresh(
        payload.refresh_token,
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def logout(
    refresh_token: str = Header(...),
    service: AuthService = Depends(get_auth_service),
):
    await service.logout(refresh_token)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/me",
    response_model=UserResponse,
)
async def me(
    current_user: User = Depends(
        get_current_active_user,
    ),
):
    return UserResponse.model_validate(current_user)