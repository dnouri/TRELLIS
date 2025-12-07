"""
Server-Sent Events (SSE) streaming utilities.

Status: Infrastructure ready - not yet integrated into endpoints.
        Currently tested via test_streaming.py.
        Integration planned for future SSE streaming endpoint.

Provides functions for creating SSE-formatted events for real-time
progress updates during long-running generation operations.

Usage (when integrated):
    @modal.web_endpoint(method="POST")
    def generate_stream(self, request: dict):
        from starlette.responses import StreamingResponse

        async def event_generator():
            yield progress_event("preprocessing", 0, "Starting...")
            # ... generation steps ...
            yield complete_event({"state": "...", "video": "..."})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

Note: Modal sends 303 redirect after 150 seconds idle (50 min effective).
Our 90-second pipeline is well within limits.
"""

from __future__ import annotations

import json
from typing import Any


def create_sse_event(
    event_type: str,
    data: dict[str, Any],
    event_id: str | None = None,
) -> str:
    """
    Create an SSE-formatted event string.

    Args:
        event_type: Event type (e.g., 'progress', 'complete', 'error')
        data: Event data to be JSON-encoded
        event_id: Optional event ID for client-side tracking

    Returns:
        SSE-formatted string ready to be sent to client
    """
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(data)}")
    return "\n".join(lines) + "\n\n"


def progress_event(stage: str, progress: float, message: str = "") -> str:
    """
    Create a progress update event.

    Args:
        stage: Current stage name (e.g., 'preprocessing', 'generation')
        progress: Progress percentage (0-100)
        message: Optional human-readable message

    Returns:
        SSE-formatted progress event
    """
    return create_sse_event(
        "progress",
        {"stage": stage, "progress": progress, "message": message},
    )


def complete_event(result: dict[str, Any]) -> str:
    """
    Create a completion event with result data.

    Args:
        result: Result data to send to client

    Returns:
        SSE-formatted completion event
    """
    return create_sse_event("complete", result)


def error_event(error_code: str, message: str) -> str:
    """
    Create an error event.

    Args:
        error_code: Machine-readable error code
        message: Human-readable error message

    Returns:
        SSE-formatted error event
    """
    return create_sse_event("error", {"code": error_code, "message": message})
