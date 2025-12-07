"""
Tests for the health check endpoint.

The health endpoint should:
- Be accessible via GET /health
- Return {"status": "ok"} for basic liveness
- Not require authentication (for load balancers)
"""


class TestHealthResponse:
    """Tests for health endpoint response structure."""

    def test_returns_status_ok(self) -> None:
        """Health response should include status: ok."""
        from trellis_modal.service.service import health_response

        response = health_response()
        assert response["status"] == "ok"

    def test_returns_dict(self) -> None:
        """Health response should be a dictionary."""
        from trellis_modal.service.service import health_response

        response = health_response()
        assert isinstance(response, dict)

    def test_includes_service_name(self) -> None:
        """Health response should identify the service."""
        from trellis_modal.service.service import health_response

        response = health_response()
        assert response["service"] == "trellis-api"
