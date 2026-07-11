from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from core.auth import create_access_token, decode_access_token, token_matches_user_state
from core.enterprise_store import EnterpriseUser


def make_user(updated_at: datetime | None) -> EnterpriseUser:
    return EnterpriseUser(
        id="7c03f4d9-4434-4632-9e74-52515db323bf",
        org_id="153b3989-8d2e-4777-bf8d-1203d75e5698",
        department_id=None,
        email="member@example.com",
        display_name="Member",
        password_hash="not-used",
        role="member",
        is_active=True,
        last_login_at=None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=updated_at,
    )


class TokenStateTests(unittest.TestCase):
    def test_token_contains_a_session_version_after_the_user_update(self) -> None:
        updated_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        token = create_access_token(SimpleNamespace(jwt_secret="unit-test-secret", jwt_ttl_minutes=10), make_user(updated_at))
        payload = decode_access_token(SimpleNamespace(jwt_secret="unit-test-secret"), token)

        self.assertIn("session_issued_at_ms", payload)
        self.assertTrue(token_matches_user_state(payload, make_user(updated_at)))

    def test_password_or_role_change_invalidates_an_older_token(self) -> None:
        updated_at = datetime.now(timezone.utc)
        payload = {"session_issued_at_ms": int((updated_at - timedelta(milliseconds=1)).timestamp() * 1000)}

        self.assertFalse(token_matches_user_state(payload, make_user(updated_at)))

    def test_legacy_token_without_a_session_version_is_rejected(self) -> None:
        self.assertFalse(token_matches_user_state({"iat": 1}, make_user(None)))
