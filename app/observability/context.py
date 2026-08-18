from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from contextvars import ContextVar, Token


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: UUID | None = None
    user_id: UUID | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    method: str | None = None
    path: str | None = None
    query_params: dict | None = None
    request_body: dict | list | str | None = None


_request_context: ContextVar[RequestContext] = ContextVar(
    "request_context",
    default=RequestContext(),
)


def get_request_context() -> RequestContext:
    return _request_context.get()


def bind_request_context(**updates) -> Token:
    current = _request_context.get()
    data = {
        "request_id": current.request_id,
        "user_id": current.user_id,
        "ip_address": current.ip_address,
        "user_agent": current.user_agent,
        "method": current.method,
        "path": current.path,
        "query_params": current.query_params,
        "request_body": current.request_body,
    }
    data.update(updates)
    return _request_context.set(RequestContext(**data))


def reset_request_context(token: Token) -> None:
    _request_context.reset(token)


def set_user_id(user_id: UUID) -> None:
    current = _request_context.get()
    _request_context.set(
        RequestContext(
            request_id=current.request_id,
            user_id=user_id,
            ip_address=current.ip_address,
            user_agent=current.user_agent,
            method=current.method,
            path=current.path,
            query_params=current.query_params,
            request_body=current.request_body,
        )
    )
