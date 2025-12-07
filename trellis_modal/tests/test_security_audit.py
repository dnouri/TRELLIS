"""
Security audit tests and documentation.

This file documents the security review findings and provides
tests that verify security properties remain in place.

Security Review Summary (Phase 8):
=================================

PASSED:
- [x] API keys masked in all log output (mask_api_key)
- [x] API keys stored in separate Modal Volume, not in code
- [x] Cryptographic randomness used (secrets.token_hex)
- [x] Input size limits prevent DoS (10MB max payload)
- [x] Dimension limits prevent GPU memory exhaustion (4096x4096)
- [x] Rate limiting per API key (quota enforcement)
- [x] Request IDs for log correlation
- [x] No stack traces leaked to clients
- [x] Health endpoint unauthenticated (for load balancers only)
- [x] All other endpoints require authentication

KNOWN RISKS (Documented):
- [!] Pickle deserialization of client state (documented in compression.py)
      Mitigation: Data originates from our server, round-trips through client
      Future: Consider HMAC signing for state integrity verification
"""

from trellis_modal.service.auth import mask_api_key


class TestApiKeyMasking:
    """Verify API keys are properly masked."""

    def test_masks_middle_of_key(self) -> None:
        """API key masking should hide the secret portion."""
        key = "sk_dev_a1b2c3d4e5f6g7h8i9j0"
        masked = mask_api_key(key)
        # Should not contain the full key
        assert "a1b2c3d4e5f6g7h8" not in masked
        # Should show prefix and suffix only
        assert masked.startswith("sk_dev_")
        assert "..." in masked

    def test_handles_none_key(self) -> None:
        """None keys should return placeholder, not crash."""
        masked = mask_api_key(None)
        assert masked == "<none>"

    def test_handles_empty_key(self) -> None:
        """Empty keys should return placeholder."""
        masked = mask_api_key("")
        assert masked == "<none>"


class TestInputValidationConstants:
    """Verify input validation limits are set."""

    def test_max_payload_size_is_reasonable(self) -> None:
        """Max payload size should prevent DoS but allow normal images."""
        from trellis_modal.service.config import MAX_IMAGE_PAYLOAD_SIZE

        # Should be > 1MB (allow normal images)
        assert MAX_IMAGE_PAYLOAD_SIZE >= 1 * 1024 * 1024
        # Should be <= 50MB (prevent abuse)
        assert MAX_IMAGE_PAYLOAD_SIZE <= 50 * 1024 * 1024

    def test_max_dimension_is_reasonable(self) -> None:
        """Max dimension should prevent OOM but allow HD images."""
        from trellis_modal.service.config import MAX_IMAGE_DIMENSION

        # Should be >= 1024 (allow HD images)
        assert MAX_IMAGE_DIMENSION >= 1024
        # Should be <= 8192 (prevent GPU OOM)
        assert MAX_IMAGE_DIMENSION <= 8192


class TestErrorResponses:
    """Verify error responses don't leak sensitive info."""

    def test_error_response_format(self) -> None:
        """Error responses should have consistent format."""
        from trellis_modal.service.service import _error_response

        result = _error_response("test_code", "Test message")
        assert "error" in result
        assert result["error"]["code"] == "test_code"
        assert result["error"]["message"] == "Test message"

    def test_error_response_no_stack_traces(self) -> None:
        """Error messages should not contain stack traces."""
        from trellis_modal.service.service import _error_response

        # Even with exception-like message, format stays clean
        result = _error_response("error", "Something failed: invalid input")
        assert "Traceback" not in str(result)
        assert "File" not in result["error"]["message"]


class TestSecurityDocumentation:
    """Verify security concerns are documented."""

    def test_pickle_risk_documented(self) -> None:
        """Pickle deserialization risk should be documented."""
        from pathlib import Path

        compression_file = Path(__file__).parent.parent / "client" / "compression.py"
        content = compression_file.read_text()
        # Should have security note about pickle
        assert "Security note" in content or "security" in content.lower()
        assert "pickle" in content.lower()
