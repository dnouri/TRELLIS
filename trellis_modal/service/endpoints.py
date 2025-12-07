"""
Modal web endpoints documentation for the TRELLIS service.

The actual endpoints are implemented on TRELLISService in service.py
to benefit from @modal.enter() hooks for efficient model loading.

Endpoint Summary:
-----------------

POST /generate
    Generate 3D model from image
    Request: {"image": "<base64>", "seed": 42, ...}
    Response: {"state": "<base64>", "video": "<base64>"}

GET /health
    Health check
    Response: {"status": "ok", "version": "1.0.0"}

Future Endpoints (not yet implemented):
----------------------------------------

POST /extract_glb
    Extract GLB mesh from state

POST /extract_gaussian
    Extract Gaussian splat from state

SSE Streaming:
--------------
For SSE streaming with progress events, a streaming version of generate
can be added using a generator function. The streaming utilities are
available in trellis_modal/service/streaming.py.

Example SSE flow:
    event: progress
    data: {"stage": "generation", "progress": 50.0, "message": "..."}

    event: complete
    data: {"state": "...", "video": "..."}

    event: error
    data: {"code": "generation_error", "message": "..."}
"""

from __future__ import annotations

# Endpoint implementations are in service.py on TRELLISService class
# This file serves as documentation and reference.
