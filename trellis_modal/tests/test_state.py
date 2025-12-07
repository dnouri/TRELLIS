"""
Tests for state serialization utilities.

Uses mock objects from conftest.py that mimic TRELLIS Gaussian and Mesh
interfaces since the real types require GPU.
"""

import numpy as np

from trellis_modal.service.state import pack_state

from .conftest import MockGaussian, MockMesh, create_mock_gaussian, create_mock_mesh


class TestPackState:
    """Tests for pack_state function."""

    def test_returns_dict_with_gaussian_key(self) -> None:
        """Result should have 'gaussian' key."""
        gs, mesh = _create_mock_state()
        result = pack_state(gs, mesh)
        assert "gaussian" in result

    def test_returns_dict_with_mesh_key(self) -> None:
        """Result should have 'mesh' key."""
        gs, mesh = _create_mock_state()
        result = pack_state(gs, mesh)
        assert "mesh" in result

    def test_gaussian_contains_init_params(self) -> None:
        """Gaussian dict should contain init_params fields."""
        gs, mesh = _create_mock_state()
        result = pack_state(gs, mesh)

        assert result["gaussian"]["aabb"] == [0, 0, 0, 1, 1, 1]
        assert result["gaussian"]["sh_degree"] == 0
        assert result["gaussian"]["scaling_activation"] == "exp"

    def test_gaussian_contains_numpy_arrays(self) -> None:
        """Gaussian dict should contain numpy arrays for tensor data."""
        gs, mesh = _create_mock_state()
        result = pack_state(gs, mesh)

        assert isinstance(result["gaussian"]["_xyz"], np.ndarray)
        assert isinstance(result["gaussian"]["_features_dc"], np.ndarray)
        assert isinstance(result["gaussian"]["_scaling"], np.ndarray)
        assert isinstance(result["gaussian"]["_rotation"], np.ndarray)
        assert isinstance(result["gaussian"]["_opacity"], np.ndarray)

    def test_mesh_contains_vertices_and_faces(self) -> None:
        """Mesh dict should contain vertices and faces arrays."""
        gs, mesh = _create_mock_state()
        result = pack_state(gs, mesh)

        assert isinstance(result["mesh"]["vertices"], np.ndarray)
        assert isinstance(result["mesh"]["faces"], np.ndarray)

    def test_preserves_array_values(self) -> None:
        """Array values should be preserved correctly."""
        gs, mesh = _create_mock_state()
        result = pack_state(gs, mesh)

        np.testing.assert_array_equal(
            result["gaussian"]["_xyz"],
            gs._xyz.numpy(),
        )
        np.testing.assert_array_equal(
            result["mesh"]["vertices"],
            mesh.vertices.numpy(),
        )

    def test_preserves_array_dtypes(self) -> None:
        """Array dtypes should be preserved."""
        gs, mesh = _create_mock_state()
        result = pack_state(gs, mesh)

        assert result["gaussian"]["_xyz"].dtype == np.float32
        assert result["mesh"]["faces"].dtype == np.int64


def _create_mock_state() -> tuple[MockGaussian, MockMesh]:
    """Create mock Gaussian and Mesh for testing."""
    return create_mock_gaussian(), create_mock_mesh()
