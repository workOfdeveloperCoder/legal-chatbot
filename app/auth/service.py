from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.refresh_token import RefreshToken
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import LoginRequest, RegisterRequest, RefreshTokenResponse
from app.core.jwt import JWTManager

from app.schemas.auth import (
    LoginResponse,
    AuthTokens,
    UserResponse,
    RefreshTokenResponse
)
from app.core.password import (
    hash_password,
    verify_password,
)
from app.models.activity_log import ActivityAction
from app.services.activity_log_service import ActivityLogService


class AuthService:
    def __init__(
        self,
        user_repository: UserRepository,
        refresh_repository: RefreshTokenRepository,
        db: AsyncSession,
        activity_log_service: ActivityLogService,
    ):
        self.db = db
        self.users = user_repository
        self.refresh_tokens = refresh_repository
        self.activity_logs = activity_log_service

    async def register(
        self,
        payload: RegisterRequest,
    ) -> User:
        email = payload.email.lower().strip()
        username = payload.username.lower().strip()

        async with self.db.begin():

            if await self.users.get_by_email(email):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already exists.",
                )

            if await self.users.get_by_username(username):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already exists.",
                )

            user = User(
                email=email,
                username=username,
                full_name=payload.full_name.strip(),
                password_hash=hash_password(payload.password),
            )

            user = await self.users.create(user)

        await self.activity_logs.record(
            action=ActivityAction.AUTH_REGISTER,
            summary=f'Registered account "{user.username}"',
            user_id=user.id,
            entity_type="user",
            entity_id=user.id,
            metadata={"email": user.email, "username": user.username},
        )

        return user


    async def login(
        self,
        payload: LoginRequest,
    ) -> LoginResponse:

        email = payload.email.lower().strip()

        async with self.db.begin():

            user = await self.users.get_by_email(email)

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials.",
                )

            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account disabled.",
                )

            if not verify_password(
                payload.password,
                user.password_hash,
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials.",
                )

            access_token = JWTManager.create_access_token(user.id)
            refresh_token = JWTManager.create_refresh_token(user.id)

            refresh = RefreshToken(
                user_id=user.id,
                token_hash=hashlib.sha256(
                    refresh_token.encode()
                ).hexdigest(),
                expires_at=datetime.now(timezone.utc)
                + timedelta(days=30),
            )

            await self.refresh_tokens.create(refresh)

            # only if this column exists
            # user.last_login_at = datetime.now(timezone.utc)

            await self.users.update(user)

        await self.activity_logs.record(
            action=ActivityAction.AUTH_LOGIN,
            summary=f'Logged in as "{user.username}"',
            user_id=user.id,
            entity_type="user",
            entity_id=user.id,
            metadata={"email": user.email},
        )

        return LoginResponse(
            user=UserResponse.model_validate(user),
            tokens=AuthTokens(
                access_token=access_token,
                refresh_token=refresh_token,
            ),
        )
    
    async def logout(
        self,
        refresh_token: str,
    ):

        token_hash = hashlib.sha256(
            refresh_token.encode()
        ).hexdigest()

        async with self.db.begin():

            token = await self.refresh_tokens.get_by_hash(
                token_hash
            )

            if token:
                token.revoked_at = datetime.now(
                    timezone.utc
                )

                await self.refresh_tokens.update(token)

                await self.activity_logs.record(
                    action=ActivityAction.AUTH_LOGOUT,
                    summary="Logged out",
                    user_id=token.user_id,
                    entity_type="user",
                    entity_id=token.user_id,
                )

    async def refresh(
        self,
        refresh_token: str,
    ) -> RefreshTokenResponse:

        token_hash = hashlib.sha256(
            refresh_token.encode()
        ).hexdigest()

        async with self.db.begin():

            stored = await self.refresh_tokens.get_by_hash(
                token_hash
            )

            if not stored:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token.",
                )

            if stored.revoked_at:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token revoked.",
                )

            if stored.expires_at < datetime.now(
                timezone.utc
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token expired.",
                )

            user = await self.users.get(stored.user_id)

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token.",
                )

            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account disabled.",
                )

            stored.revoked_at = datetime.now(
                timezone.utc
            )

            await self.refresh_tokens.update(stored)

            access_token = JWTManager.create_access_token(
                user.id
            )
            new_refresh_token = (
                JWTManager.create_refresh_token(user.id)
            )

            await self.refresh_tokens.create(
                RefreshToken(
                    user_id=user.id,
                    token_hash=hashlib.sha256(
                        new_refresh_token.encode()
                    ).hexdigest(),
                    expires_at=datetime.now(timezone.utc)
                    + timedelta(days=30),
                )
            )

        await self.activity_logs.record(
            action=ActivityAction.AUTH_TOKEN_REFRESH,
            summary=f'Refreshed access token for "{user.username}"',
            user_id=user.id,
            entity_type="user",
            entity_id=user.id,
        )

        return RefreshTokenResponse(
            tokens=AuthTokens(
                access_token=access_token,
                refresh_token=new_refresh_token,
            )
        )