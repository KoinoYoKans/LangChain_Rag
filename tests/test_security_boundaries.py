from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from core.ingestion import validate_public_http_url
from core.upload_safety import sanitize_filename, validate_document_bytes


class UploadSafetyTests(unittest.TestCase):
    def test_filename_is_reduced_to_a_safe_basename(self) -> None:
        self.assertEqual(sanitize_filename("..\\..\\finance/plan.pdf"), "plan.pdf")
        self.assertEqual(sanitize_filename("\x00"), "uploaded")

    def test_pdf_signature_and_binary_text_are_checked(self) -> None:
        with self.assertRaises(HTTPException) as invalid_pdf:
            validate_document_bytes("report.pdf", b"not a pdf", 1024)
        self.assertEqual(invalid_pdf.exception.status_code, 400)

        with self.assertRaises(HTTPException) as binary_text:
            validate_document_bytes("notes.txt", b"text\x00binary", 1024)
        self.assertEqual(binary_text.exception.status_code, 400)


class UrlImportSafetyTests(unittest.TestCase):
    @staticmethod
    def public_dns_answer(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    def test_accepts_a_public_https_url_and_removes_fragment(self) -> None:
        with patch("core.ingestion.socket.getaddrinfo", self.public_dns_answer):
            self.assertEqual(
                validate_public_http_url("https://example.com/docs?q=rag#heading"),
                "https://example.com/docs?q=rag",
            )

    def test_rejects_private_dns_answers_and_mismatched_ports(self) -> None:
        private_answer = [(2, 1, 6, "", ("10.0.0.8", 443))]
        with patch("core.ingestion.socket.getaddrinfo", return_value=private_answer):
            with self.assertRaisesRegex(ValueError, "Private, loopback"):
                validate_public_http_url("https://example.com")

        with self.assertRaisesRegex(ValueError, "port must match"):
            validate_public_http_url("https://example.com:80")

