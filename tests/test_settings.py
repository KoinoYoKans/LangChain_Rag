from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.middleware.cors import CORSMiddleware

from config.settings import AppSettings
from core.main import create_app, resolve_client_ip


class SettingsValidationTests(unittest.TestCase):
    def test_cors_origin_allowlist_must_contain_origins(self) -> None:
        env = {
            "CORS_ALLOWED_ORIGINS": "https://console.example.com,http://localhost:5173",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = AppSettings.load(env_file="/tmp/does-not-exist")
        self.assertEqual(
            settings.cors_allowed_origins,
            ("https://console.example.com", "http://localhost:5173"),
        )
        self.assertNotIn("CORS_ALLOWED_ORIGINS contains an invalid origin: *", settings.validation_errors())

    def test_rejects_wildcard_cors_origin(self) -> None:
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "*"}, clear=False):
            settings = AppSettings.load(env_file="/tmp/does-not-exist")
        self.assertIn("CORS_ALLOWED_ORIGINS contains an invalid origin: *", settings.validation_errors())

    def test_rejects_cors_origin_with_path(self) -> None:
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "https://console.example.com/app"}, clear=False):
            settings = AppSettings.load(env_file="/tmp/does-not-exist")
        self.assertIn(
            "CORS_ALLOWED_ORIGINS contains an invalid origin: https://console.example.com/app",
            settings.validation_errors(),
        )

    def test_invalid_cors_origin_is_not_registered_by_the_application(self) -> None:
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "*"}, clear=False):
            app = create_app()
        cors_middleware = [item for item in app.user_middleware if item.cls is CORSMiddleware]
        self.assertEqual(cors_middleware, [])

    def test_proxy_header_is_used_only_for_a_trusted_peer(self) -> None:
        self.assertEqual(
            resolve_client_ip("172.30.0.10", "203.0.113.7, 172.30.0.10", ("172.30.0.0/24",)),
            "203.0.113.7",
        )
        self.assertEqual(
            resolve_client_ip("172.31.0.10", "203.0.113.7", ("172.30.0.0/24",)),
            "172.31.0.10",
        )
