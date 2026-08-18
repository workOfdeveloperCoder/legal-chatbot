from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

from app.core.config import settings

ALGORITHM = "HS256"


class JWTManager:
    @staticmethod
    def _build_payload(
        subject: UUID | str,
        token_type: str,
        expires_delta: timedelta,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)

        return {
            "sub": str(subject),
            "type": token_type,
            "iat": now,
            "nbf": now,
            "exp": now + expires_delta,
            "iss": settings.APP_NAME,
        }

    @classmethod
    def create_access_token(
        cls,
        subject: UUID | str,
    ) -> str:
        payload = cls._build_payload(
            subject=subject,
            token_type="access",
            expires_delta=timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            ),
        )

        return jwt.encode(
            payload,
            settings.SECRET_KEY,
            algorithm=ALGORITHM,
        )

    @classmethod
    def create_refresh_token(
        cls,
        subject: UUID | str,
    ) -> str:
        payload = cls._build_payload(
            subject=subject,
            token_type="refresh",
            expires_delta=timedelta(
                days=settings.REFRESH_TOKEN_EXPIRE_DAYS,
            ),
        )

        return jwt.encode(
            payload,
            settings.SECRET_KEY,
            algorithm=ALGORITHM,
        )

    @staticmethod
    def decode(
        token: str,
    ) -> dict[str, Any]:
        try:
            return jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[ALGORITHM],
                issuer=settings.APP_NAME,
            )

        except ExpiredSignatureError:
            raise

        except InvalidTokenError:
            raise

    @staticmethod
    def decode_access_token(token: str) -> dict[str, Any]:
        payload = JWTManager.decode(token)

        if payload.get("type") != "access":
            raise InvalidTokenError("Invalid access token.")

        return payload

    @staticmethod
    def decode_refresh_token(token: str) -> dict[str, Any]:
        payload = JWTManager.decode(token)

        if payload.get("type") != "refresh":
            raise InvalidTokenError("Invalid refresh token.")

        return payload