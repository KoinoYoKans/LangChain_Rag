from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from fastapi import Header, HTTPException, status

from config.settings import AppSettings
from core.enterprise_store import EnterpriseUser, get_user_by_id


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return f"pbkdf2_sha256${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, salt_text, digest_text = password_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    salt = base64.urlsafe_b64decode(salt_text.encode())
    expected = base64.urlsafe_b64decode(digest_text.encode())
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return hmac.compare_digest(actual, expected)


def create_access_token(settings: AppSettings, user: EnterpriseUser) -> str:
    now_ms = int(time.time() * 1000)
    user_updated_at_ms = datetime_to_epoch_ms(user.updated_at)
    session_issued_at_ms = max(now_ms, user_updated_at_ms or 0)
    now = session_issued_at_ms // 1000
    payload = {
        "sub": user.id,
        "org_id": user.org_id,
        "role": user.role,
        "iat": now,
        "session_issued_at_ms": session_issued_at_ms,
        "exp": now + settings.jwt_ttl_minutes * 60,
    }
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_b64_json(header)}.{_b64_json(payload)}"
    signature = hmac.new(settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(signature)}"


def decode_access_token(settings: AppSettings, token: str) -> dict[str, object]:
    try:
        header_text, payload_text, signature_text = token.split(".", 2)
        signing_input = f"{header_text}.{payload_text}"
        expected = hmac.new(settings.jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        actual = _b64_decode(signature_text)
        if not hmac.compare_digest(actual, expected):
            raise ValueError("invalid signature")
        payload = json.loads(_b64_decode(payload_text))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    return payload


@dataclass(frozen=True)
class CurrentUser:
    id: str
    org_id: str
    department_id: str | None
    email: str
    display_name: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_manager(self) -> bool:
        return self.role in {"admin", "manager"}


async def require_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    settings = AppSettings.load()
    payload = decode_access_token(settings, authorization.split(" ", 1)[1].strip())
    user_id = str(payload.get("sub") or "")
    try:
        UUID(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject") from exc
    user = await get_user_by_id(settings, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User is inactive or missing")
    if not token_matches_user_state(payload, user):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked; sign in again")
    return CurrentUser(
        id=user.id,
        org_id=user.org_id,
        department_id=user.department_id,
        email=user.email,
        display_name=user.display_name,
        role=user.role,
    )


def _b64_json(value: dict[str, object]) -> str:
    return _b64(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _b64_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def token_matches_user_state(payload: dict[str, object], user: EnterpriseUser) -> bool:
    """Invalidate all prior sessions whenever a user identity record changes."""
    issued_at_ms = payload.get("session_issued_at_ms")
    if not isinstance(issued_at_ms, int) or issued_at_ms <= 0:
        return False
    user_updated_at_ms = datetime_to_epoch_ms(user.updated_at)
    return user_updated_at_ms is None or user_updated_at_ms <= issued_at_ms


def datetime_to_epoch_ms(value: datetime | None) -> int | None:
    return int(value.timestamp() * 1000) if value else None
