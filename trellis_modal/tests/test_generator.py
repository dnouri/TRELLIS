"""
Tests for TRELLISGenerator with mock pipeline injection.

These tests verify the generator's behavior without loading real models,
using the constructor injection pattern established in Fix #5.
"""

from typing import Any

import numpy as np
import pytest
from PIL import Image

from trellis_modal.service.generator import TRELLISGenerator

from .conftest import (
    create_mock_gaussian,
    create_mock_mesh,
)


def _create_mock_outputs() -> dict[str, list]:
    """Create mock pipeline outputs."""
    return {"gaussian": [create_mock_gaussian()], "mesh": [create_mock_mesh()]}


class MockPipeline:
    """Mock pipeline that mimics the real TrellisImageTo3DPipeline interface."""

    def __init__(self) -> None:
        self.models = {
            "sparse_structure_decoder": "mock",
            "sparse_structure_flow_model": "mock",
            "slat_decoder_gs": "mock",
            "slat_decoder_rf": "mock",
            "slat_decoder_mesh": "mock",
            "slat_flow_model": "mock",
            "image_cond_model": "mock",
        }
        self._on_gpu = False
        self._last_run_params: dict[str, Any] | None = None

    def cuda(self) -> "MockPipeline":
        self._on_gpu = True
        return self

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Mock preprocessing - just returns the image."""
        return image

    def run(
        self,
        image: Image.Image,
        seed: int,
        formats: list[str],
        preprocess_image: bool,
        sparse_structure_sampler_params: dict[str, Any],
        slat_sampler_params: dict[str, Any],
    ) -> dict[str, list]:
        """Mock run - stores params and returns mock outputs."""
        self._last_run_params = {
            "seed": seed,
            "formats": formats,
            "preprocess_image": preprocess_image,
            "sparse_structure_sampler_params": sparse_structure_sampler_params,
            "slat_sampler_params": slat_sampler_params,
        }
        return _create_mock_outputs()


def mock_pipeline_factory(model_name: str) -> MockPipeline:
    """Factory that creates mock pipelines."""
    assert model_name == "JeffreyXiang/TRELLIS-image-large"
    return MockPipeline()


class TestTRELLISGeneratorWithMock:
    """Test TRELLISGenerator with mock pipeline injection."""

    def test_load_model_cpu_uses_factory(self) -> None:
        """Generator should use injected factory to create pipeline."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)

        generator.load_model_cpu()

        assert generator.pipeline is not None
        assert isinstance(generator.pipeline, MockPipeline)
        assert generator.load_time_cpu > 0

    def test_move_model_gpu_calls_cuda(self) -> None:
        """move_model_gpu should call cuda() on the pipeline."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_model_cpu()

        generator.move_model_gpu()

        assert generator.pipeline._on_gpu is True
        assert generator.load_time_gpu > 0

    def test_is_loaded_property(self) -> None:
        """is_loaded should reflect pipeline state."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)

        assert generator.is_loaded is False

        generator.load_model_cpu()
        assert generator.is_loaded is True

    def test_total_load_time_property(self) -> None:
        """total_load_time should sum CPU and GPU times."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_model_cpu()
        generator.move_model_gpu()

        assert generator.total_load_time == (
            generator.load_time_cpu + generator.load_time_gpu
        )

    def test_load_models_convenience_method(self) -> None:
        """load_models should load to CPU then move to GPU."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)

        generator.load_models()

        assert generator.is_loaded is True
        assert generator.pipeline._on_gpu is True

    def test_move_model_gpu_without_load_raises(self) -> None:
        """move_model_gpu should raise if pipeline not loaded."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)

        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            generator.move_model_gpu()

    def test_models_property_accessible(self) -> None:
        """pipeline.models should be accessible after loading."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_model_cpu()

        models = generator.pipeline.models
        assert "sparse_structure_decoder" in models
        assert len(models) == 7


class TestTRELLISGeneratorErrorHandling:
    """Test error handling in TRELLISGenerator."""

    def test_load_model_cpu_wraps_exception(self) -> None:
        """load_model_cpu should wrap exceptions with context."""

        def failing_factory(model_name: str) -> MockPipeline:
            raise ValueError("Network error")

        generator = TRELLISGenerator(pipeline_factory=failing_factory)

        with pytest.raises(RuntimeError, match="Failed to load TRELLIS model"):
            generator.load_model_cpu()

    def test_move_model_gpu_wraps_exception(self) -> None:
        """move_model_gpu should wrap exceptions with context."""

        class FailingPipeline(MockPipeline):
            def cuda(self) -> "FailingPipeline":
                raise RuntimeError("CUDA OOM")

        def failing_cuda_factory(model_name: str) -> FailingPipeline:
            return FailingPipeline()

        generator = TRELLISGenerator(pipeline_factory=failing_cuda_factory)
        generator.load_model_cpu()

        with pytest.raises(RuntimeError, match="Failed to move TRELLIS model to GPU"):
            generator.move_model_gpu()


class TestGenerate3D:
    """Tests for generate_3d method."""

    def test_generate_3d_returns_dict_with_gaussian_key(self) -> None:
        """Result should contain gaussian data."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        image = Image.new("RGB", (512, 512), color="red")

        result = generator.generate_3d(
            image=image,
            seed=42,
            ss_sampling_steps=12,
            slat_sampling_steps=12,
            slat_guidance_strength=3.0,
        )

        assert "gaussian" in result

    def test_generate_3d_returns_dict_with_mesh_key(self) -> None:
        """Result should contain mesh data."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        image = Image.new("RGB", (512, 512), color="red")

        result = generator.generate_3d(
            image=image,
            seed=42,
            ss_sampling_steps=12,
            slat_sampling_steps=12,
            slat_guidance_strength=3.0,
        )

        assert "mesh" in result

    def test_generate_3d_gaussian_contains_numpy_arrays(self) -> None:
        """Gaussian data should contain numpy arrays."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        image = Image.new("RGB", (512, 512), color="red")

        result = generator.generate_3d(
            image=image,
            seed=42,
            ss_sampling_steps=12,
            slat_sampling_steps=12,
            slat_guidance_strength=3.0,
        )

        assert isinstance(result["gaussian"]["_xyz"], np.ndarray)

    def test_generate_3d_passes_correct_params_to_pipeline(self) -> None:
        """Pipeline.run should receive correct parameters."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        image = Image.new("RGB", (512, 512), color="red")

        generator.generate_3d(
            image=image,
            seed=123,
            ss_sampling_steps=8,
            slat_sampling_steps=16,
            slat_guidance_strength=5.0,
        )

        params = generator.pipeline._last_run_params
        assert params["seed"] == 123
        assert params["sparse_structure_sampler_params"]["steps"] == 8
        assert params["slat_sampler_params"]["steps"] == 16
        assert params["slat_sampler_params"]["cfg_strength"] == 5.0

    def test_generate_3d_raises_if_not_loaded(self) -> None:
        """generate_3d should raise if pipeline not loaded."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        image = Image.new("RGB", (512, 512), color="red")

        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            generator.generate_3d(
                image=image,
                seed=42,
                ss_sampling_steps=12,
                slat_sampling_steps=12,
                slat_guidance_strength=3.0,
            )

    def test_generate_3d_preprocesses_image(self) -> None:
        """generate_3d should preprocess the input image."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        image = Image.new("RGB", (512, 512), color="red")

        generator.generate_3d(
            image=image,
            seed=42,
            ss_sampling_steps=12,
            slat_sampling_steps=12,
            slat_guidance_strength=3.0,
        )

        # Pipeline run should be called with preprocess_image=False
        # because we preprocess separately
        params = generator.pipeline._last_run_params
        assert params["preprocess_image"] is False


def _torch_available() -> bool:
    """Check if torch is available for testing."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


class TestRenderPreviewVideo:
    """Tests for render_preview_video method."""

    def test_raises_if_not_loaded(self) -> None:
        """render_preview_video should raise if pipeline not loaded."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        state = _create_packed_state()

        # This test works without torch because the check happens before imports
        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            generator.render_preview_video(state)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch required for video rendering tests",
    )
    def test_returns_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """render_preview_video should return bytes (MP4 data)."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        state = _create_packed_state()

        # Mock render_utils to avoid GPU dependency
        _mock_render_utils(monkeypatch)

        result = generator.render_preview_video(state)
        assert isinstance(result, bytes)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch required for video rendering tests",
    )
    def test_returns_non_empty_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Video bytes should not be empty."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        state = _create_packed_state()

        _mock_render_utils(monkeypatch)

        result = generator.render_preview_video(state)
        assert len(result) > 0


def _create_packed_state() -> dict[str, Any]:
    """Create a packed state dict for testing."""
    return {
        "gaussian": {
            "aabb": [0, 0, 0, 1, 1, 1],
            "sh_degree": 0,
            "mininum_kernel_size": 0.0,
            "scaling_bias": 0.01,
            "opacity_bias": 0.1,
            "scaling_activation": "exp",
            "_xyz": np.random.randn(100, 3).astype(np.float32),
            "_features_dc": np.random.randn(100, 3).astype(np.float32),
            "_scaling": np.random.randn(100, 3).astype(np.float32),
            "_rotation": np.random.randn(100, 4).astype(np.float32),
            "_opacity": np.random.randn(100, 1).astype(np.float32),
        },
        "mesh": {
            "vertices": np.random.randn(50, 3).astype(np.float32),
            "faces": np.arange(150).reshape(50, 3).astype(np.int64),
        },
    }


def _mock_render_utils(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock TRELLIS render_utils for testing without GPU."""
    # Create fake video frames (4 frames of 64x64 RGB)
    fake_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]

    def mock_render_video(obj: Any, num_frames: int = 4) -> dict[str, list]:
        # Return fake frames matching TRELLIS interface
        return {"color": fake_frames, "normal": fake_frames}

    # Patch the render_utils module that will be imported in generator
    import sys

    mock_module = type(sys)("mock_render_utils")
    mock_module.render_video = mock_render_video
    monkeypatch.setitem(sys.modules, "trellis.utils.render_utils", mock_module)


class MockTrimesh:
    """Mock trimesh.Trimesh that exports GLB data."""

    def __init__(self) -> None:
        self._export_called = False

    def export(self, file_obj: Any, file_type: str = "glb") -> None:
        """Write fake GLB data to file object."""
        self._export_called = True
        # GLB magic header is 'glTF' followed by version and length
        fake_glb = b"glTF" + b"\x02\x00\x00\x00" + b"\x00" * 100
        file_obj.write(fake_glb)


def _mock_postprocessing_utils(monkeypatch: pytest.MonkeyPatch) -> MockTrimesh:
    """Mock TRELLIS postprocessing_utils for testing without GPU."""
    import sys

    mock_mesh = MockTrimesh()

    def mock_to_glb(
        app_rep: Any,
        mesh: Any,
        simplify: float = 0.95,
        texture_size: int = 1024,
        verbose: bool = False,
    ) -> MockTrimesh:
        return mock_mesh

    mock_module = type(sys)("mock_postprocessing_utils")
    mock_module.to_glb = mock_to_glb
    monkeypatch.setitem(sys.modules, "trellis.utils.postprocessing_utils", mock_module)

    return mock_mesh


class TestExtractGLB:
    """Tests for extract_glb method."""

    def test_raises_if_not_loaded(self) -> None:
        """extract_glb should raise if pipeline not loaded."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        state = _create_packed_state()

        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            generator.extract_glb(state, mesh_simplify_ratio=0.95, texture_size=1024)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch required for extract_glb tests",
    )
    def test_returns_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """extract_glb should return bytes (GLB data)."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        state = _create_packed_state()

        _mock_postprocessing_utils(monkeypatch)

        result = generator.extract_glb(
            state, mesh_simplify_ratio=0.95, texture_size=1024
        )
        assert isinstance(result, bytes)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch required for extract_glb tests",
    )
    def test_returns_non_empty_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GLB bytes should not be empty."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        state = _create_packed_state()

        _mock_postprocessing_utils(monkeypatch)

        result = generator.extract_glb(
            state, mesh_simplify_ratio=0.95, texture_size=1024
        )
        assert len(result) > 0

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch required for extract_glb tests",
    )
    def test_glb_starts_with_magic_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GLB should start with glTF magic header."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        state = _create_packed_state()

        _mock_postprocessing_utils(monkeypatch)

        result = generator.extract_glb(
            state, mesh_simplify_ratio=0.95, texture_size=1024
        )
        assert result[:4] == b"glTF"


class TestExtractGaussian:
    """Tests for extract_gaussian method."""

    def test_raises_if_not_loaded(self) -> None:
        """extract_gaussian should raise if pipeline not loaded."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        state = _create_packed_state()

        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            generator.extract_gaussian(state)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch required for extract_gaussian tests",
    )
    def test_returns_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """extract_gaussian should return bytes (PLY data)."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        state = _create_packed_state()

        _mock_gaussian_save_ply(monkeypatch)

        result = generator.extract_gaussian(state)
        assert isinstance(result, bytes)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch required for extract_gaussian tests",
    )
    def test_returns_non_empty_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PLY bytes should not be empty."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        state = _create_packed_state()

        _mock_gaussian_save_ply(monkeypatch)

        result = generator.extract_gaussian(state)
        assert len(result) > 0

    @pytest.mark.skipif(
        not _torch_available(),
        reason="torch required for extract_gaussian tests",
    )
    def test_ply_starts_with_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PLY should start with 'ply' header."""
        generator = TRELLISGenerator(pipeline_factory=mock_pipeline_factory)
        generator.load_models()
        state = _create_packed_state()

        _mock_gaussian_save_ply(monkeypatch)

        result = generator.extract_gaussian(state)
        assert result[:3] == b"ply"


def _mock_gaussian_save_ply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the Gaussian.save_ply method for testing without GPU."""
    import sys

    import torch

    # Create a mock Gaussian class that has save_ply method
    class MockGaussianRepr:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def save_ply(self, file_obj: Any) -> None:
            """Write fake PLY data to file object."""
            fake_ply = b"ply\nformat binary_little_endian 1.0\n" + b"\x00" * 100
            file_obj.write(fake_ply)

    # Mock the unpack_state to return our mock
    def mock_unpack_state(state: dict) -> tuple:
        from easydict import EasyDict as edict

        gs = MockGaussianRepr()
        mesh = edict(
            vertices=torch.zeros(10, 3),
            faces=torch.zeros(10, 3, dtype=torch.long),
        )
        return gs, mesh

    # Patch the state module
    mock_state_module = type(sys)("mock_state")
    mock_state_module.unpack_state = mock_unpack_state
    monkeypatch.setitem(sys.modules, "modal_service.state", mock_state_module)
