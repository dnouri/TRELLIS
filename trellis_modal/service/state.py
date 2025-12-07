"""
State serialization utilities for TRELLIS generation results.

Converts between TRELLIS representation objects (Gaussian, Mesh) and
plain dictionaries with numpy arrays for serialization and compression.

These functions run server-side where TRELLIS types are available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trellis.representations import Gaussian, MeshExtractResult


def pack_state(gaussian: Gaussian, mesh: MeshExtractResult) -> dict[str, Any]:
    """
    Pack Gaussian and Mesh objects into a serializable dictionary.

    Converts GPU tensors to CPU numpy arrays for efficient serialization.
    The dictionary structure matches TRELLIS app.py format for compatibility.

    Args:
        gaussian: Gaussian splat representation from generation
        mesh: Mesh extraction result from generation

    Returns:
        Dictionary with 'gaussian' and 'mesh' keys containing numpy arrays
    """
    return {
        "gaussian": {
            **gaussian.init_params,
            "_xyz": gaussian._xyz.cpu().numpy(),
            "_features_dc": gaussian._features_dc.cpu().numpy(),
            "_scaling": gaussian._scaling.cpu().numpy(),
            "_rotation": gaussian._rotation.cpu().numpy(),
            "_opacity": gaussian._opacity.cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.cpu().numpy(),
            "faces": mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict[str, Any]) -> tuple[Gaussian, Any]:
    """
    Unpack a serialized state dictionary into Gaussian and Mesh objects.

    Converts numpy arrays back to GPU tensors and reconstructs TRELLIS
    representation objects.

    Args:
        state: Dictionary from pack_state with 'gaussian' and 'mesh' keys

    Returns:
        Tuple of (Gaussian, mesh_edict) where mesh_edict has vertices/faces
    """
    import torch
    from easydict import EasyDict as edict
    from trellis.representations import Gaussian

    gs_data = state["gaussian"]

    gs = Gaussian(
        aabb=gs_data["aabb"],
        sh_degree=gs_data["sh_degree"],
        mininum_kernel_size=gs_data["mininum_kernel_size"],
        scaling_bias=gs_data["scaling_bias"],
        opacity_bias=gs_data["opacity_bias"],
        scaling_activation=gs_data["scaling_activation"],
    )

    gs._xyz = torch.tensor(gs_data["_xyz"], device="cuda")
    gs._features_dc = torch.tensor(gs_data["_features_dc"], device="cuda")
    gs._scaling = torch.tensor(gs_data["_scaling"], device="cuda")
    gs._rotation = torch.tensor(gs_data["_rotation"], device="cuda")
    gs._opacity = torch.tensor(gs_data["_opacity"], device="cuda")

    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device="cuda"),
        faces=torch.tensor(state["mesh"]["faces"], device="cuda"),
    )

    return gs, mesh
