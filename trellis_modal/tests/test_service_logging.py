"""
Tests for service logging utilities.

Tests request ID generation and structured logging functions
used in production hardening.
"""

import logging

import pytest


class TestGenerateRequestId:
    """Tests for request ID generation."""

    def test_returns_string(self) -> None:
        """Request ID should be a string."""
        from trellis_modal.service.service import generate_request_id

        request_id = generate_request_id()
        assert isinstance(request_id, str)

    def test_has_prefix(self) -> None:
        """Request ID should have 'req_' prefix."""
        from trellis_modal.service.service import generate_request_id

        request_id = generate_request_id()
        assert request_id.startswith("req_")

    def test_is_unique(self) -> None:
        """Each call should produce unique ID."""
        from trellis_modal.service.service import generate_request_id

        ids = [generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_reasonable_length(self) -> None:
        """Request ID should be reasonable length for logging."""
        from trellis_modal.service.service import generate_request_id

        request_id = generate_request_id()
        # req_ prefix (4) + 16 hex chars = 20 total
        assert 15 <= len(request_id) <= 40


class TestLogRequest:
    """Tests for _log_request function."""

    def test_includes_request_id_in_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Log output should include request_id when provided."""
        from trellis_modal.service.service import _log_request

        with caplog.at_level(logging.INFO):
            _log_request(
                endpoint="test",
                api_key="sk_dev_test123",
                duration_ms=100.0,
                status="success",
                request_id="req_abc123",
            )

        assert "req_abc123" in caplog.text

    def test_includes_all_fields(self, caplog: pytest.LogCaptureFixture) -> None:
        """Log should include endpoint, duration, status."""
        from trellis_modal.service.service import _log_request

        with caplog.at_level(logging.INFO):
            _log_request(
                endpoint="generate",
                api_key="sk_dev_test",
                duration_ms=1234.56,
                status="success",
                request_id="req_test",
            )

        # Check structured log data is present
        assert "generate" in caplog.text
        assert "1234.56" in caplog.text
        assert "success" in caplog.text

    def test_logs_error_at_warning_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Error status should log at WARNING level."""
        from trellis_modal.service.service import _log_request

        with caplog.at_level(logging.WARNING):
            _log_request(
                endpoint="generate",
                api_key="sk_dev_test",
                duration_ms=500.0,
                status="error",
                error_code="cuda_oom",
                request_id="req_err",
            )

        assert "Request failed" in caplog.text
        assert "cuda_oom" in caplog.text

    def test_works_without_request_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should still work if request_id is None (backwards compat)."""
        from trellis_modal.service.service import _log_request

        with caplog.at_level(logging.INFO):
            _log_request(
                endpoint="test",
                api_key="sk_dev_test",
                duration_ms=100.0,
                status="success",
            )

        assert "Request completed" in caplog.text

    def test_includes_extra_metrics(self, caplog: pytest.LogCaptureFixture) -> None:
        """Extra metrics should be included in log output."""
        from trellis_modal.service.service import _log_request

        with caplog.at_level(logging.INFO):
            _log_request(
                endpoint="generate",
                api_key="sk_dev_test",
                duration_ms=100.0,
                status="success",
                request_id="req_test",
                extra_metrics={"output_size_bytes": 12345, "image_width": 512},
            )

        assert "12345" in caplog.text
        assert "512" in caplog.text

    def test_extra_metrics_are_flat(self, caplog: pytest.LogCaptureFixture) -> None:
        """Extra metrics should be merged into log_data, not nested."""
        from trellis_modal.service.service import _log_request

        with caplog.at_level(logging.INFO):
            _log_request(
                endpoint="generate",
                api_key="sk_dev_test",
                duration_ms=100.0,
                status="success",
                extra_metrics={"custom_field": "test_value"},
            )

        # The field should appear directly, not as nested "extra_metrics"
        assert "custom_field" in caplog.text
        assert "test_value" in caplog.text
