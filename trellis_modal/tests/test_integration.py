"""
Integration tests for the TRELLIS Modal service.

This module contains two types of tests:

1. Local integration tests (no GPU required):
   - Cross-module wiring verification
   - State roundtrip through compression
   - SSE event format consistency
   - Auth flow with file I/O

2. E2E integration tests (require deployed Modal service):
   - Marked with @pytest.mark.integration
   - Skipped by default (run with: pytest -m integration)
   - Require TRELLIS_API_URL and TRELLIS_API_KEY environment variables
"""

import json

import numpy as np
import pytest

from trellis_modal.client.compression import compress_state, decompress_state
from trellis_modal.service.auth import check_auth, save_keys
from trellis_modal.service.state import pack_state
from trellis_modal.service.streaming import (
    complete_event,
    create_sse_event,
    error_event,
    progress_event,
)

from trellis_modal.tests.conftest import create_mock_gaussian, create_mock_mesh


class TestLocalIntegration:
    """Local integration tests that verify cross-module wiring without GPU."""

    def test_state_pack_compress_decompress_roundtrip(self) -> None:
        """State packed by server should survive compression roundtrip on client."""
        # Simulate server: create mock Gaussian and Mesh objects
        mock_gaussian = create_mock_gaussian()
        mock_mesh = create_mock_mesh()

        # Server packs state
        packed = pack_state(mock_gaussian, mock_mesh)

        # Client compresses for storage
        compressed = compress_state(packed)
        assert len(compressed) > 0

        # Client decompresses when sending back
        decompressed = decompress_state(compressed)

        # Verify data integrity
        assert decompressed["gaussian"]["aabb"] == [0, 0, 0, 1, 1, 1]
        assert decompressed["gaussian"]["sh_degree"] == 0
        np.testing.assert_array_equal(
            decompressed["gaussian"]["_xyz"], packed["gaussian"]["_xyz"]
        )
        np.testing.assert_array_equal(
            decompressed["mesh"]["vertices"], packed["mesh"]["vertices"]
        )

    def test_sse_event_format_parseable(self) -> None:
        """SSE events created by server should be parseable by standard SSE parsers."""
        # Server creates events
        events = [
            progress_event("preprocessing", 50.0, "Processing image..."),
            complete_event({"state": "abc123", "video": "xyz789"}),
            error_event("cuda_oom", "GPU out of memory"),
        ]

        for event_str in events:
            # Verify SSE format: lines separated by \n, ends with \n\n
            assert event_str.endswith("\n\n"), "SSE event must end with double newline"

            lines = event_str.rstrip("\n").split("\n")
            assert len(lines) >= 2, "SSE event must have at least event and data lines"

            # Parse event type
            event_line = next(line for line in lines if line.startswith("event:"))
            event_type = event_line.split(": ", 1)[1]
            assert event_type in ["progress", "complete", "error"]

            # Parse data
            data_line = next(line for line in lines if line.startswith("data:"))
            data_json = data_line.split(": ", 1)[1]
            data = json.loads(data_json)
            assert isinstance(data, dict)

    def test_auth_check_with_file_storage(self, tmp_path) -> None:
        """Full auth flow: create keys file, validate key, reject invalid."""
        keys_file = tmp_path / "keys.json"

        # Create keys file with one valid key
        keys_data = {
            "version": 1,
            "keys": {
                "sk_dev_validkey123": {"name": "test", "active": True, "usage_count": 0}
            },
        }
        save_keys(str(keys_file), keys_data)

        # Valid key should pass
        is_valid, info = check_auth("sk_dev_validkey123", str(keys_file))
        assert is_valid is True
        assert info["name"] == "test"

        # Invalid key should fail
        is_valid, info = check_auth("sk_dev_wrongkey", str(keys_file))
        assert is_valid is False
        assert info is None

        # No key should fail
        is_valid, info = check_auth("", str(keys_file))
        assert is_valid is False

    def test_sse_event_id_tracking(self) -> None:
        """SSE events with IDs should include id field for client-side tracking."""
        event = create_sse_event("progress", {"step": 5}, event_id="evt_123")

        assert "id: evt_123" in event
        assert "event: progress" in event
        assert "data:" in event


@pytest.mark.integration
class TestGenerationWorkflow:
    """E2E integration tests for the generation workflow.

    These tests require a deployed Modal service with GPU.
    Run with: pytest -m integration

    Environment variables required:
    - TRELLIS_API_URL: Base URL of deployed Modal service
    - TRELLIS_API_KEY: Valid API key for authentication
    """

    def test_health_check(self) -> None:
        """Service should respond to health check."""
        pytest.skip("Requires deployed Modal service")

    def test_generate_returns_valid_state(self) -> None:
        """Generate endpoint should return valid compressed state."""
        pytest.skip("Requires deployed Modal service")

    def test_extract_glb_produces_valid_file(self) -> None:
        """Extract GLB should produce valid GLB file."""
        pytest.skip("Requires deployed Modal service")

    def test_extract_gaussian_produces_valid_file(self) -> None:
        """Extract Gaussian should produce valid PLY file."""
        pytest.skip("Requires deployed Modal service")

    def test_invalid_api_key_rejected(self) -> None:
        """Request with invalid API key should be rejected."""
        pytest.skip("Requires deployed Modal service")


@pytest.mark.integration
class TestSSEStreaming:
    """E2E integration tests for SSE streaming.

    These tests require a deployed Modal service with GPU.
    """

    def test_progress_events_received(self) -> None:
        """Client should receive progress events during generation."""
        pytest.skip("Requires deployed Modal service")

    def test_error_events_formatted_correctly(self) -> None:
        """Error responses should be properly formatted SSE events."""
        pytest.skip("Requires deployed Modal service")
