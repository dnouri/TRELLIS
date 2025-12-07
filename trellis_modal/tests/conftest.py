"""
Shared test fixtures and mock objects for TRELLIS tests.

These mocks mimic TRELLIS types that require GPU, allowing tests
to run without CUDA dependencies.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


class MockTensor:
    """Mock tensor that mimics PyTorch tensor interface."""

    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def cpu(self) -> "MockTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._data


@dataclass
class MockGaussian:
    """Mock Gaussian that mimics TRELLIS Gaussian interface."""

    init_params: dict[str, Any]
    _xyz: MockTensor
    _features_dc: MockTensor
    _scaling: MockTensor
    _rotation: MockTensor
    _opacity: MockTensor


@dataclass
class MockMesh:
    """Mock Mesh that mimics TRELLIS MeshExtractResult interface."""

    vertices: MockTensor
    faces: MockTensor


def create_mock_gaussian() -> MockGaussian:
    """Create a MockGaussian with random data."""
    init_params = {
        "aabb": [0, 0, 0, 1, 1, 1],
        "sh_degree": 0,
        "mininum_kernel_size": 0.0,
        "scaling_bias": 0.01,
        "opacity_bias": 0.1,
        "scaling_activation": "exp",
    }
    return MockGaussian(
        init_params=init_params,
        _xyz=MockTensor(np.random.randn(100, 3).astype(np.float32)),
        _features_dc=MockTensor(np.random.randn(100, 3).astype(np.float32)),
        _scaling=MockTensor(np.random.randn(100, 3).astype(np.float32)),
        _rotation=MockTensor(np.random.randn(100, 4).astype(np.float32)),
        _opacity=MockTensor(np.random.randn(100, 1).astype(np.float32)),
    )


def create_mock_mesh() -> MockMesh:
    """Create a MockMesh with random data."""
    return MockMesh(
        vertices=MockTensor(np.random.randn(50, 3).astype(np.float32)),
        faces=MockTensor(np.arange(150).reshape(50, 3).astype(np.int64)),
    )
