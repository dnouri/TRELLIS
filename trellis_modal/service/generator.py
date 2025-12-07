"""
TRELLIS Generator class for Modal deployment.

This module contains the main TRELLISGenerator class that handles:
- Model loading and GPU initialization
- Image-to-3D generation pipeline
- Mesh and Gaussian extraction
- Memory snapshots for fast cold starts

Note on memory snapshots:
- GPU is NOT available in snap=True unless enable_gpu_snapshot is enabled
- Recommended pattern: load to CPU in snap=True, move to GPU in snap=False
- Modal retries failed containers aggressively (12+ times)

Note on imports:
- TRELLIS imports (trellis.*) are deferred because they only work inside
  the Modal container where /opt/TRELLIS is on PYTHONPATH.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from PIL import Image

from .config import MODEL_NAME


class PipelineProtocol(Protocol):
    """Protocol defining the pipeline interface for type checking and mocking."""

    models: dict[str, Any]

    def cuda(self) -> "PipelineProtocol": ...

    def preprocess_image(self, image: "Image.Image") -> "Image.Image": ...

    def run(
        self,
        image: "Image.Image",
        seed: int,
        formats: list[str],
        preprocess_image: bool,
        sparse_structure_sampler_params: dict[str, Any],
        slat_sampler_params: dict[str, Any],
    ) -> dict[str, list[Any]]: ...


class PipelineFactory(Protocol):
    """Factory protocol for creating pipelines."""

    def __call__(self, model_name: str) -> PipelineProtocol: ...


def _default_pipeline_factory(model_name: str) -> PipelineProtocol:
    """Default factory that loads the real TRELLIS pipeline."""
    from trellis.pipelines import TrellisImageTo3DPipeline

    return TrellisImageTo3DPipeline.from_pretrained(model_name)


class TRELLISGenerator:
    """
    GPU-accelerated TRELLIS generator deployed on Modal.

    Uses Modal's two-phase @enter pattern for optimal cold starts:
    1. @modal.enter(snap=True) - Load model to CPU (captured in snapshot)
    2. @modal.enter(snap=False) - Move model to GPU (runs on each restore)

    Requires enable_memory_snapshot=True on the Modal class decorator.

    Args:
        pipeline_factory: Optional factory for creating pipelines. Defaults to
            loading the real TRELLIS pipeline. Inject a mock for testing.

    Timing data (from spike on A100-40GB):
    - from_pretrained: ~63s (with HF cache warm)
    - .cuda(): ~0.5-4.5s
    - VRAM after load: 5.73 GB
    - 7 models: sparse_structure_decoder, sparse_structure_flow_model,
      slat_decoder_gs, slat_decoder_rf, slat_decoder_mesh,
      slat_flow_model, image_cond_model
    """

    def __init__(
        self,
        pipeline_factory: Callable[[str], PipelineProtocol] | None = None,
    ) -> None:
        """Initialize generator state. Models loaded in enter methods."""
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self.pipeline: PipelineProtocol | None = None
        self.load_time_cpu: float = 0.0
        self.load_time_gpu: float = 0.0

    def load_model_cpu(self) -> None:
        """
        Load model to CPU memory. Decorated with @modal.enter(snap=True).

        This runs BEFORE snapshot creation and is captured in the saved state.
        CPU-only operations here - GPU not available in snap=True.
        """
        start = time.perf_counter()
        try:
            self.pipeline = self._pipeline_factory(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TRELLIS model '{MODEL_NAME}' to CPU. "
                f"Check HuggingFace cache and network connectivity."
            ) from e
        self.load_time_cpu = time.perf_counter() - start

    def move_model_gpu(self) -> None:
        """
        Move model to GPU. Decorated with @modal.enter(snap=False).

        This runs AFTER snapshot restore on every container start.
        GPU is available here. Moves CPU-loaded model to CUDA.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_model_cpu() first.")

        start = time.perf_counter()
        try:
            self.pipeline.cuda()
        except Exception as e:
            raise RuntimeError(
                "Failed to move TRELLIS model to GPU. "
                "Check CUDA availability and GPU memory."
            ) from e
        self.load_time_gpu = time.perf_counter() - start

    def load_models(self) -> None:
        """
        Load models in single phase (CPU + GPU). Use with @modal.enter().

        Convenience method for when memory snapshots are not enabled.
        Combines load_model_cpu() and move_model_gpu() into one call.
        """
        self.load_model_cpu()
        self.move_model_gpu()

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return self.pipeline is not None

    @property
    def total_load_time(self) -> float:
        """Total model loading time in seconds."""
        return self.load_time_cpu + self.load_time_gpu

    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory if CUDA is available."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch not available (testing environment)

    def generate_3d(
        self,
        image: Image.Image,
        seed: int,
        ss_sampling_steps: int,
        slat_sampling_steps: int,
        slat_guidance_strength: float,
    ) -> dict[str, Any]:
        """
        Generate 3D representation from input image.

        Args:
            image: Input PIL Image (will be preprocessed)
            seed: Random seed for reproducibility
            ss_sampling_steps: Sparse structure sampling steps
            slat_sampling_steps: SLAT sampling steps
            slat_guidance_strength: Classifier-free guidance strength

        Returns:
            Dictionary containing Gaussian and mesh state (packed for serialization)
        """
        from .state import pack_state

        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")

        processed_image = self.pipeline.preprocess_image(image)

        outputs = self.pipeline.run(
            processed_image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

        state = pack_state(outputs["gaussian"][0], outputs["mesh"][0])

        # Clean up GPU memory after generation
        self._cleanup_gpu_memory()

        return state

    def render_preview_video(
        self,
        state: dict[str, Any],
        num_frames: int = 120,
        fps: int = 15,
    ) -> bytes:
        """
        Render a preview video from generation state.

        Renders both Gaussian splat (color) and mesh (normals) side by side.

        Args:
            state: Packed state from generate_3d
            num_frames: Number of frames to render
            fps: Frames per second for output video

        Returns:
            MP4 video as bytes
        """
        # Check pipeline first (before any GPU-dependent imports)
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")

        import imageio
        import numpy as np
        from trellis.utils import render_utils

        from .state import unpack_state

        gaussian, mesh = unpack_state(state)

        # Render Gaussian splat as color video
        video_color = render_utils.render_video(gaussian, num_frames=num_frames)[
            "color"
        ]
        # Render mesh as normal map video
        video_normal = render_utils.render_video(mesh, num_frames=num_frames)["normal"]

        # Concatenate side by side
        video_frames = [
            np.concatenate([video_color[i], video_normal[i]], axis=1)
            for i in range(len(video_color))
        ]

        # Encode to MP4 in memory using imageio's <bytes> magic URI
        video_bytes = imageio.mimwrite("<bytes>", video_frames, format="mp4", fps=fps)

        # Clean up GPU memory after rendering
        self._cleanup_gpu_memory()

        return video_bytes

    def extract_glb(
        self,
        state: dict[str, Any],
        mesh_simplify_ratio: float,
        texture_size: int,
    ) -> bytes:
        """
        Extract GLB mesh from generation state.

        Performs mesh simplification, hole filling, UV parametrization,
        and texture baking to produce a GLB file.

        Args:
            state: Packed state from generate_3d (gaussian + mesh dict)
            mesh_simplify_ratio: Mesh simplification ratio (0-1, higher = more faces)
            texture_size: Texture resolution in pixels

        Returns:
            GLB file contents as bytes
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")

        import io

        from trellis.utils import postprocessing_utils

        from .state import unpack_state

        gaussian, mesh = unpack_state(state)

        glb = postprocessing_utils.to_glb(
            gaussian,
            mesh,
            simplify=mesh_simplify_ratio,
            texture_size=texture_size,
            verbose=False,
        )

        buffer = io.BytesIO()
        glb.export(buffer, file_type="glb")
        buffer.seek(0)
        glb_bytes = buffer.read()

        # Clean up GPU memory after extraction
        self._cleanup_gpu_memory()

        return glb_bytes

    def extract_gaussian(self, state: dict[str, Any]) -> bytes:
        """
        Extract Gaussian splat from generation state.

        Exports the Gaussian splat representation as a PLY file.
        This is much faster than GLB extraction since no mesh processing needed.

        Args:
            state: Packed state from generate_3d (gaussian + mesh dict)

        Returns:
            PLY file contents as bytes
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_models() first.")

        import io

        from .state import unpack_state

        gaussian, _ = unpack_state(state)

        buffer = io.BytesIO()
        gaussian.save_ply(buffer)
        buffer.seek(0)
        ply_bytes = buffer.read()

        # Clean up GPU memory after extraction
        self._cleanup_gpu_memory()

        return ply_bytes
