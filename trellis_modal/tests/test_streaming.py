"""
Tests for SSE streaming utilities.

Tests the Server-Sent Events formatting functions that create
properly formatted SSE events for client consumption.
"""

import json
import pytest
from trellis_modal.service.streaming import (
    create_sse_event,
    progress_event,
    complete_event,
    error_event,
)


class TestCreateSSEEvent:
    """Tests for create_sse_event function."""

    def test_event_starts_with_event_type(self) -> None:
        """Event should start with 'event: type' line."""
        event = create_sse_event("progress", {"stage": "encoding"})
        assert event.startswith("event: progress\n")

    def test_event_contains_data_line(self) -> None:
        """Event should contain 'data: json' line."""
        event = create_sse_event("progress", {"stage": "encoding"})
        assert "\ndata: " in event

    def test_event_ends_with_double_newline(self) -> None:
        """Event must end with double newline (SSE terminator)."""
        event = create_sse_event("progress", {"stage": "encoding"})
        assert event.endswith("\n\n")

    def test_data_is_valid_json(self) -> None:
        """Data portion should be valid JSON."""
        event = create_sse_event("test", {"key": "value", "num": 42})
        # Extract data line
        for line in event.split("\n"):
            if line.startswith("data: "):
                data_json = line[6:]  # Remove "data: " prefix
                parsed = json.loads(data_json)
                assert parsed == {"key": "value", "num": 42}
                return
        pytest.fail("No data line found in event")

    def test_event_with_id(self) -> None:
        """Event with ID should include id line."""
        event = create_sse_event("progress", {"x": 1}, event_id="evt-123")
        assert "id: evt-123\n" in event

    def test_event_without_id(self) -> None:
        """Event without ID should not include id line."""
        event = create_sse_event("progress", {"x": 1})
        assert "id:" not in event


class TestProgressEvent:
    """Tests for progress_event function."""

    def test_progress_event_type(self) -> None:
        """Progress event should have type 'progress'."""
        event = progress_event("encoding", 50.0, "Encoding image...")
        assert event.startswith("event: progress\n")

    def test_progress_event_contains_stage(self) -> None:
        """Progress event should include stage in data."""
        event = progress_event("sparse_structure", 25.0)
        data = _extract_data(event)
        assert data["stage"] == "sparse_structure"

    def test_progress_event_contains_progress(self) -> None:
        """Progress event should include progress percentage."""
        event = progress_event("encoding", 75.5)
        data = _extract_data(event)
        assert data["progress"] == 75.5

    def test_progress_event_contains_message(self) -> None:
        """Progress event should include message when provided."""
        event = progress_event("decoding", 100.0, "Decoding complete!")
        data = _extract_data(event)
        assert data["message"] == "Decoding complete!"

    def test_progress_event_empty_message(self) -> None:
        """Progress event with empty message should still include key."""
        event = progress_event("decoding", 100.0, "")
        data = _extract_data(event)
        assert data["message"] == ""


class TestCompleteEvent:
    """Tests for complete_event function."""

    def test_complete_event_type(self) -> None:
        """Complete event should have type 'complete'."""
        event = complete_event({"state": "abc123"})
        assert event.startswith("event: complete\n")

    def test_complete_event_contains_result(self) -> None:
        """Complete event should include result data."""
        result = {"state": "compressed_data", "video": "base64_video"}
        event = complete_event(result)
        data = _extract_data(event)
        assert data == result


class TestErrorEvent:
    """Tests for error_event function."""

    def test_error_event_type(self) -> None:
        """Error event should have type 'error'."""
        event = error_event("cuda_oom", "Out of GPU memory")
        assert event.startswith("event: error\n")

    def test_error_event_contains_code(self) -> None:
        """Error event should include error code."""
        event = error_event("validation_error", "Invalid image format")
        data = _extract_data(event)
        assert data["code"] == "validation_error"

    def test_error_event_contains_message(self) -> None:
        """Error event should include error message."""
        event = error_event("timeout", "Request timed out after 300s")
        data = _extract_data(event)
        assert data["message"] == "Request timed out after 300s"


def _extract_data(event: str) -> dict:
    """Helper to extract and parse data from SSE event."""
    for line in event.split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise ValueError("No data line found in event")
