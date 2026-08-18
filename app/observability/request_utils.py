from __future__ import annotations

from uuid import UUID

from jwt import InvalidTokenError

from app.core.jwt import JWTManager


def extract_user_id_from_authorization(
    authorization: str | None,
) -> UUID | None:
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        return None

    try:
        payload = JWTManager.decode_access_token(token)
    except InvalidTokenError:
        return None

    user_id = payload.get("sub")

    if not user_id:
        return None

    try:
        return UUID(str(user_id))
    except ValueError:
        return None


def get_client_ip(
    *,
    x_forwarded_for: str | None,
    x_real_ip: str | None,
    client_host: str | None,
) -> str | None:
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    if x_real_ip:
        return x_real_ip.strip()

    return client_host
