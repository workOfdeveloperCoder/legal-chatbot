from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.user import UserRole


# ---------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=255)
    username: str = Field(min_length=3, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    full_name: str
    username: str
    email: EmailStr

    role: UserRole

    is_active: bool
    is_verified: bool


class AuthTokens(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"


class LoginResponse(BaseModel):
    user: UserResponse
    tokens: AuthTokens


class RegisterResponse(BaseModel):
    user: UserResponse


class RefreshTokenResponse(BaseModel):
    tokens: AuthTokens