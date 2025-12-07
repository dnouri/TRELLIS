"""
Modal service for TRELLIS 3D generation.

This module defines the Modal class that wraps TRELLISGenerator,
providing the @modal.enter() lifecycle hooks and @modal.method()
endpoints for the generation pipeline.

Usage:
    modal run -m trellis_modal.service.service::test_service   # Test health check
    modal deploy -m trellis_modal.service.service              # Deploy with snapshots

Benchmark Results (A100-SXM4-40GB, 2025-12-07):
===============================================
Cold start (first call, DINOv2 baked into image):
  - Total time: ~150s
  - CPU model load: ~43s
  - GPU transfer: ~1.3s

Warm container (subsequent calls):
  - Response time: ~0.35s
  - Speedup: ~425x vs cold start

Memory usage:
  - VRAM after load: 5.73 GB
  - Total GPU memory: 42.41 GB

GPU Snapshots:
  - Only work with `modal deploy`, not `modal run`
  - Message "Memory snapshots are disabled for ephemeral apps" is expected
  - Warm containers already fast (0.31s) even without snapshots
  - For production, keep containers warm with scaledown_window=300
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import modal
from fastapi import Request

from .auth import check_rate_limit, increment_usage, load_keys, mask_api_key, validate_api_key
from .config import (
    API_KEYS_PATH,
    GPU_MEMORY_SNAPSHOT,
    GPU_TYPE,
    HF_CACHE_PATH,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PAYLOAD_SIZE,
)
from .generator import TRELLISGenerator
from .image import api_keys_volume, app, hf_cache_volume, trellis_image

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """
    Generate a unique request ID for request tracing.

    Returns:
        String in format 'req_' + 16 hex characters (e.g., 'req_a1b2c3d4e5f6g7h8')
    """
    return f"req_{secrets.token_hex(8)}"

# Volume mount path for API keys (parent directory of API_KEYS_PATH)
API_KEYS_VOLUME_PATH = "/data"

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class GenerateParams:
    """Validated parameters for the generate endpoint."""

    image: Image.Image
    seed: int
    ss_sampling_steps: int
    slat_sampling_steps: int
    slat_guidance_strength: float


@dataclass
class ExtractGLBParams:
    """Validated parameters for the extract_glb endpoint."""

    state: dict
    mesh_simplify_ratio: float
    texture_size: int


def _error_response(code: str, message: str) -> dict:
    """Create a standardized error response."""
    return {"error": {"code": code, "message": message}}


def health_response() -> dict:
    """
    Create a standardized health check response.

    Returns a simple response for load balancer health checks.
    No authentication required.

    Returns:
        Dictionary with status and service identification.
    """
    return {"status": "ok", "service": "trellis-api"}


def _log_request(
    endpoint: str,
    api_key: str | None,
    duration_ms: float,
    status: str,
    error_code: str | None = None,
    request_id: str | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    """
    Log structured request completion.

    Args:
        endpoint: Name of the endpoint (e.g., "generate", "extract_glb")
        api_key: API key (will be masked in logs)
        duration_ms: Request duration in milliseconds
        status: "success" or "error"
        error_code: Error code if status is "error"
        request_id: Unique request ID for correlation
        extra_metrics: Additional metrics to include (e.g., output_size_bytes)
    """
    log_data: dict[str, Any] = {
        "endpoint": endpoint,
        "api_key": mask_api_key(api_key),
        "duration_ms": round(duration_ms, 2),
        "status": status,
    }
    if request_id:
        log_data["request_id"] = request_id
    if error_code:
        log_data["error_code"] = error_code
    if extra_metrics:
        log_data.update(extra_metrics)

    if status == "success":
        logger.info("Request completed: %s", log_data)
    else:
        logger.warning("Request failed: %s", log_data)


def _check_auth_or_error(
    http_request: Request,
) -> tuple[bool, dict[str, Any] | None] | dict:
    """
    Check API key authentication from request headers.

    Also increments usage counter for valid keys (in memory only, not persisted).
    Usage is logged for audit trail.

    Args:
        http_request: FastAPI Request object

    Returns:
        If valid: (True, key_info dict)
        If invalid: error response dict
    """
    api_key = http_request.headers.get("X-API-Key")
    keys_data = load_keys(API_KEYS_PATH)
    is_valid, key_info = validate_api_key(api_key, keys_data)

    if not is_valid:
        logger.warning("Auth failed for key: %s", mask_api_key(api_key))
        return _error_response("unauthorized", "Invalid or missing API key")

    # Check rate limit before allowing request
    if not check_rate_limit(key_info):
        logger.warning(
            "Rate limit exceeded for key: %s (quota: %d)",
            mask_api_key(api_key),
            key_info.get("quota", 0),
        )
        return _error_response("rate_limited", "Rate limit exceeded. Quota exhausted.")

    # Track usage (in memory only - not persisted to avoid latency)
    increment_usage(api_key, keys_data)
    usage_count = keys_data["keys"][api_key].get("usage_count", 0)

    logger.info(
        "Auth success for key: %s (usage: %d)",
        mask_api_key(api_key),
        usage_count,
    )
    return is_valid, key_info


def _parse_generate_request(request: dict) -> GenerateParams | dict:
    """
    Parse and validate a generate request.

    Args:
        request: Raw request dictionary from client

    Returns:
        GenerateParams if valid, or error dict if validation fails
    """
    import base64
    from io import BytesIO

    from PIL import Image

    if not isinstance(request, dict):
        return _error_response("validation_error", "Request must be JSON object")

    image_b64 = request.get("image")
    if not image_b64:
        return _error_response("validation_error", "Missing 'image' field")

    # Check payload size before decoding (base64 is ~33% larger than binary)
    # Use 1.4x multiplier to estimate decoded size from base64 length
    estimated_size = len(image_b64) * 3 // 4
    if estimated_size > MAX_IMAGE_PAYLOAD_SIZE:
        return _error_response(
            "validation_error",
            f"Image size exceeds limit ({estimated_size // (1024*1024)}MB > "
            f"{MAX_IMAGE_PAYLOAD_SIZE // (1024*1024)}MB)",
        )

    # Parse numeric parameters
    try:
        seed = int(request.get("seed", 42))
        ss_sampling_steps = int(request.get("ss_sampling_steps", 12))
        slat_sampling_steps = int(request.get("slat_sampling_steps", 12))
        slat_guidance_strength = float(request.get("slat_guidance_strength", 3.0))
    except (TypeError, ValueError) as e:
        return _error_response("validation_error", f"Invalid parameter: {e}")

    # Decode image
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        return _error_response("validation_error", f"Invalid image: {e}")

    # Validate image dimensions
    width, height = image.size
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        return _error_response(
            "validation_error",
            f"Image dimensions exceed limit ({width}x{height} > {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})",
        )

    return GenerateParams(
        image=image,
        seed=seed,
        ss_sampling_steps=ss_sampling_steps,
        slat_sampling_steps=slat_sampling_steps,
        slat_guidance_strength=slat_guidance_strength,
    )


def _parse_extract_glb_request(request: dict) -> ExtractGLBParams | dict:
    """
    Parse and validate an extract_glb request.

    Args:
        request: Raw request dictionary from client

    Returns:
        ExtractGLBParams if valid, or error dict if validation fails
    """
    import base64

    from trellis_modal.client.compression import decompress_state

    if not isinstance(request, dict):
        return _error_response("validation_error", "Request must be JSON object")

    state_b64 = request.get("state")
    if not state_b64:
        return _error_response("validation_error", "Missing 'state' field")

    # Parse numeric parameters
    try:
        mesh_simplify_ratio = float(request.get("mesh_simplify_ratio", 0.95))
        texture_size = int(request.get("texture_size", 1024))
    except (TypeError, ValueError) as e:
        return _error_response("validation_error", f"Invalid parameter: {e}")

    # Validate parameter ranges
    if not (0.0 < mesh_simplify_ratio <= 1.0):
        return _error_response(
            "validation_error",
            "mesh_simplify_ratio must be between 0 and 1",
        )
    if texture_size not in (512, 1024, 2048):
        return _error_response(
            "validation_error",
            "texture_size must be 512, 1024, or 2048",
        )

    # Decode and decompress state
    try:
        compressed_state = base64.b64decode(state_b64)
        state = decompress_state(compressed_state)
    except Exception as e:
        return _error_response("validation_error", f"Invalid state: {e}")

    return ExtractGLBParams(
        state=state,
        mesh_simplify_ratio=mesh_simplify_ratio,
        texture_size=texture_size,
    )


@app.cls(
    image=trellis_image,
    gpu=GPU_TYPE,
    volumes={
        HF_CACHE_PATH: hf_cache_volume,
        API_KEYS_VOLUME_PATH: api_keys_volume,
    },
    timeout=600,
    scaledown_window=300,
    enable_memory_snapshot=GPU_MEMORY_SNAPSHOT,
)
class TRELLISService:
    """
    Modal class wrapping TRELLISGenerator.

    Uses composition to separate Modal infrastructure (decorators, lifecycle)
    from domain logic (model loading, generation).

    GPU Snapshot Strategy:
    - snap=True: Load model to CPU (captured in snapshot)
    - snap=False: Move model to GPU (runs on each restore)

    If snapshots are disabled or fail, falls back to single-phase loading.
    """

    @modal.enter(snap=True)
    def load_model_to_cpu(self) -> None:
        """
        Load model to CPU memory (captured in memory snapshot).

        This runs BEFORE snapshot creation. GPU is not available here
        unless enable_gpu_snapshot is also enabled.
        """
        import time

        import torch

        # Record wall clock time - used in snap=False to detect snapshot restore
        self._snap_true_timestamp = time.time()

        self.generator = TRELLISGenerator()

        logger.info("Loading model to CPU...")
        logger.info("GPU available: %s", torch.cuda.is_available())

        self.generator.load_model_cpu()

        logger.info("CPU load complete: %.2fs", self.generator.load_time_cpu)
        logger.info("Models: %s", list(self.generator.pipeline.models.keys()))

    @modal.enter(snap=False)
    def move_model_to_gpu(self) -> None:
        """
        Move model to GPU (runs after snapshot restore).

        This runs AFTER snapshot restore on every container start.
        GPU is available here.
        """
        import time

        import torch

        # Validate that snap=True ran (or snapshot was restored)
        if not hasattr(self, "generator") or self.generator is None:
            raise RuntimeError(
                "Generator not initialized. This indicates that snap=True failed "
                "and no snapshot was restored. Check container logs for errors "
                "during model loading, or verify enable_memory_snapshot is True."
            )

        if not self.generator.is_loaded:
            raise RuntimeError(
                "Pipeline not loaded. snap=True may have failed during "
                "model loading. Check HuggingFace cache and network connectivity."
            )

        logger.info("Moving model to GPU...")
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

        self.generator.move_model_gpu()

        # Timing heuristic for snapshot detection:
        # - Cold start: snap=True ran just now, so timestamp is recent (<60s ago)
        # - Snapshot restore: snap=True ran during snapshot creation (hours/days ago)
        # Threshold of 120s accounts for slow cold starts
        time_since_snap_true = time.time() - self._snap_true_timestamp
        self._snapshot_restored = time_since_snap_true > 120.0

        logger.info("GPU transfer complete: %.2fs", self.generator.load_time_gpu)
        logger.info("VRAM: %.2f GB", torch.cuda.memory_allocated() / 1e9)
        logger.info("Time since snap=True: %.1fs", time_since_snap_true)
        logger.info("Restored from snapshot: %s", self._snapshot_restored)

    @modal.method()
    def health_check(self) -> dict:
        """
        Return health status and diagnostic information.

        Returns:
            Dictionary with status, GPU info, memory usage, and timing data.
        """
        import torch

        return {
            "status": "healthy" if self.generator.is_loaded else "unhealthy",
            "gpu": torch.cuda.get_device_name(0),
            "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "vram_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            ),
            "models_loaded": (
                list(self.generator.pipeline.models.keys())
                if self.generator.pipeline
                else []
            ),
            "load_time_cpu_seconds": round(self.generator.load_time_cpu, 2),
            "load_time_gpu_seconds": round(self.generator.load_time_gpu, 2),
            "total_load_time_seconds": round(self.generator.total_load_time, 2),
            "snapshot_restored": self._snapshot_restored,
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """
        GET /health - Health check endpoint for load balancers.

        No authentication required. Returns simple liveness status.

        Returns:
            Dict with status and service name.
        """
        return health_response()

    @modal.fastapi_endpoint(method="POST")
    def generate(self, http_request: Request, request: dict) -> dict:
        """
        POST /generate - Generate 3D model from image.

        This is a synchronous endpoint that returns all at once.
        For SSE streaming, use generate_stream().

        Requires X-API-Key header for authentication.

        Args:
            http_request: FastAPI Request for header access
            request: Dict with:
                - image: Base64-encoded input image
                - seed: Random seed (default: 42)
                - ss_sampling_steps: Sparse structure steps (default: 12)
                - slat_sampling_steps: SLAT steps (default: 12)
                - slat_guidance_strength: Guidance (default: 3.0)

        Returns:
            Dict with state (compressed, base64) and video (base64)
            On error: {"error": {"code": "...", "message": "..."}}
        """
        import base64

        import torch

        from trellis_modal.client.compression import compress_state

        request_id = generate_request_id()
        start_time = time.perf_counter()
        api_key = http_request.headers.get("X-API-Key")

        # Check authentication
        auth_result = _check_auth_or_error(http_request)
        if isinstance(auth_result, dict):  # Error response
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("generate", api_key, duration_ms, "error", "unauthorized", request_id)
            return auth_result

        # Parse and validate request
        params = _parse_generate_request(request)
        if isinstance(params, dict):  # Error response
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("generate", api_key, duration_ms, "error", "validation_error", request_id)
            return params

        # Generate 3D
        try:
            state = self.generator.generate_3d(
                image=params.image,
                seed=params.seed,
                ss_sampling_steps=params.ss_sampling_steps,
                slat_sampling_steps=params.slat_sampling_steps,
                slat_guidance_strength=params.slat_guidance_strength,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("generate", api_key, duration_ms, "error", "cuda_oom", request_id)
            return _error_response(
                "cuda_oom",
                "GPU out of memory. Try a smaller image or reduce sampling steps.",
            )
        except Exception as e:
            logger.exception("Generation failed")
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("generate", api_key, duration_ms, "error", "generation_error", request_id)
            return _error_response("generation_error", f"Generation failed: {e}")

        # Render video preview
        try:
            video_bytes = self.generator.render_preview_video(state)
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("generate", api_key, duration_ms, "error", "cuda_oom", request_id)
            return _error_response(
                "cuda_oom",
                "GPU out of memory during video rendering. Generation succeeded but video unavailable.",
            )
        except Exception as e:
            logger.exception("Video rendering failed")
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("generate", api_key, duration_ms, "error", "rendering_error", request_id)
            return _error_response("rendering_error", f"Video rendering failed: {e}")

        # Compress state
        try:
            compressed_state = compress_state(state)
            state_b64 = base64.b64encode(compressed_state).decode("utf-8")
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("generate", api_key, duration_ms, "error", "compression_error", request_id)
            return _error_response("compression_error", f"State compression failed: {e}")

        duration_ms = (time.perf_counter() - start_time) * 1000
        _log_request(
            "generate",
            api_key,
            duration_ms,
            "success",
            request_id=request_id,
            extra_metrics={
                "state_size_bytes": len(compressed_state),
                "video_size_bytes": len(video_bytes),
            },
        )
        return {
            "state": state_b64,
            "video": video_b64,
            "request_id": request_id,
        }

    @modal.fastapi_endpoint(method="POST")
    def extract_glb(self, http_request: Request, request: dict) -> dict:
        """
        POST /extract_glb - Extract GLB mesh from generation state.

        Takes compressed state from generate() and produces a GLB file.
        Requires X-API-Key header for authentication.

        Args:
            http_request: FastAPI Request for header access
            request: Dict with:
                - state: Base64-encoded compressed state from generate()
                - mesh_simplify_ratio: Simplification ratio 0-1 (default: 0.95)
                - texture_size: Texture resolution 512/1024/2048 (default: 1024)

        Returns:
            Dict with glb (base64-encoded GLB file)
            On error: {"error": {"code": "...", "message": "..."}}
        """
        import base64

        import torch

        request_id = generate_request_id()
        start_time = time.perf_counter()
        api_key = http_request.headers.get("X-API-Key")

        # Check authentication
        auth_result = _check_auth_or_error(http_request)
        if isinstance(auth_result, dict):  # Error response
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_glb", api_key, duration_ms, "error", "unauthorized", request_id)
            return auth_result

        # Parse and validate request
        params = _parse_extract_glb_request(request)
        if isinstance(params, dict):  # Error response
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_glb", api_key, duration_ms, "error", "validation_error", request_id)
            return params

        # Extract GLB
        try:
            glb_bytes = self.generator.extract_glb(
                state=params.state,
                mesh_simplify_ratio=params.mesh_simplify_ratio,
                texture_size=params.texture_size,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_glb", api_key, duration_ms, "error", "cuda_oom", request_id)
            return _error_response(
                "cuda_oom",
                "GPU out of memory during GLB extraction. Try reducing texture size.",
            )
        except Exception as e:
            logger.exception("GLB extraction failed")
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_glb", api_key, duration_ms, "error", "extraction_error", request_id)
            return _error_response("extraction_error", f"GLB extraction failed: {e}")

        # Encode result
        glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")

        duration_ms = (time.perf_counter() - start_time) * 1000
        _log_request(
            "extract_glb",
            api_key,
            duration_ms,
            "success",
            request_id=request_id,
            extra_metrics={"glb_size_bytes": len(glb_bytes)},
        )
        return {"glb": glb_b64, "request_id": request_id}

    @modal.fastapi_endpoint(method="POST")
    def extract_gaussian(self, http_request: Request, request: dict) -> dict:
        """
        POST /extract_gaussian - Extract Gaussian PLY from generation state.

        Takes compressed state from generate() and produces a PLY file.
        Much faster than GLB extraction (~1s vs ~60s).
        Requires X-API-Key header for authentication.

        Args:
            http_request: FastAPI Request for header access
            request: Dict with:
                - state: Base64-encoded compressed state from generate()

        Returns:
            Dict with ply (base64-encoded PLY file)
            On error: {"error": {"code": "...", "message": "..."}}
        """
        import base64

        import torch

        from trellis_modal.client.compression import decompress_state

        request_id = generate_request_id()
        start_time = time.perf_counter()
        api_key = http_request.headers.get("X-API-Key")

        # Check authentication
        auth_result = _check_auth_or_error(http_request)
        if isinstance(auth_result, dict):  # Error response
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_gaussian", api_key, duration_ms, "error", "unauthorized", request_id)
            return auth_result

        if not isinstance(request, dict):
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_gaussian", api_key, duration_ms, "error", "validation_error", request_id)
            return _error_response("validation_error", "Request must be JSON object")

        state_b64 = request.get("state")
        if not state_b64:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_gaussian", api_key, duration_ms, "error", "validation_error", request_id)
            return _error_response("validation_error", "Missing 'state' field")

        # Decode and decompress state
        try:
            compressed_state = base64.b64decode(state_b64)
            state = decompress_state(compressed_state)
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_gaussian", api_key, duration_ms, "error", "validation_error", request_id)
            return _error_response("validation_error", f"Invalid state: {e}")

        # Extract Gaussian PLY
        try:
            ply_bytes = self.generator.extract_gaussian(state=state)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_gaussian", api_key, duration_ms, "error", "cuda_oom", request_id)
            return _error_response(
                "cuda_oom",
                "GPU out of memory during Gaussian extraction.",
            )
        except Exception as e:
            logger.exception("Gaussian extraction failed")
            duration_ms = (time.perf_counter() - start_time) * 1000
            _log_request("extract_gaussian", api_key, duration_ms, "error", "extraction_error", request_id)
            return _error_response("extraction_error", f"Gaussian extraction failed: {e}")

        # Encode result
        ply_b64 = base64.b64encode(ply_bytes).decode("utf-8")

        duration_ms = (time.perf_counter() - start_time) * 1000
        _log_request(
            "extract_gaussian",
            api_key,
            duration_ms,
            "success",
            request_id=request_id,
            extra_metrics={"ply_size_bytes": len(ply_bytes)},
        )
        return {"ply": ply_b64, "request_id": request_id}


@app.local_entrypoint()
def test_service():
    """Test the service with a health check."""
    import json
    import time

    print("\n" + "=" * 60)
    print("TRELLIS Modal Service - Health Check")
    print("=" * 60 + "\n")

    service = TRELLISService()

    print("Starting first call (cold start)...")
    start = time.time()
    result = service.health_check.remote()
    first_call_time = time.time() - start

    print(f"\nFirst call completed in {first_call_time:.2f}s")
    print(f"Result: {json.dumps(result, indent=2)}")

    print("\nStarting second call (warm container)...")
    start = time.time()
    result = service.health_check.remote()
    second_call_time = time.time() - start

    print(f"\nSecond call completed in {second_call_time:.2f}s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cold start:     {first_call_time:.2f}s")
    print(f"Warm container: {second_call_time:.2f}s")
    print(f"Speedup:        {first_call_time / second_call_time:.1f}x")

    if result["snapshot_restored"]:
        print("\nGPU snapshots: ACTIVE")
    else:
        print("\nGPU snapshots: NOT ACTIVE (expected with modal run)")
