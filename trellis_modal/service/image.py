"""
Modal image definition for TRELLIS.

This module defines the container image with all dependencies pre-installed,
including CUDA extensions that are compiled at image build time rather than
runtime. This ensures reproducible builds and eliminates cold start compilation.

Build notes (from spike validation):
- gpu="T4" in run_commands creates a FRESH build context
- ALL dependencies (torch, wheel, setuptools) must be in the GPU block
- --no-build-isolation needed for packages requiring torch at build time
- clang required for CUDA extension builds
"""

import modal

# Constants duplicated from config.py - Modal copies this file to /root/image.py
# without the modal_service package, so imports fail. Keep in sync manually.
# Verified by tests/test_config_consistency.py
GPU_TYPE = "A100-40GB"
HF_CACHE_PATH = "/cache/huggingface"
MODEL_NAME = "JeffreyXiang/TRELLIS-image-large"

# Pinned git commits for reproducible builds
# These commits are validated to work with PyTorch 2.4.0 / CUDA 11.8
PINNED_COMMITS = {
    "nvdiffrast": "c5caf7bdb8a2448acc491a9faa47753972edd380",  # June 2023 stable
    "diffoctreerast": "b09c20b84ec3aace4729e6e18a613112320eca3a",  # TRELLIS release (Dec 2024)
    "mip_splatting": "e3a68d04f19e324b79f2b1e4caa723c59370e751",  # Pre-TRELLIS (Oct 2024)
    "trellis": "442aa1e1afb9014e80681d3bf604e8d728a86ee7",  # Latest stable (Nov 2025)
    "utils3d": "9a4eb15e4021b67b12c460c7057d642626897ec8",  # TRELLIS integration
}

# Modal app for TRELLIS
app = modal.App("trellis-3d")

# Modal volumes for persistent storage
hf_cache_volume = modal.Volume.from_name("trellis-hf-cache", create_if_missing=True)
api_keys_volume = modal.Volume.from_name("trellis-api-keys", create_if_missing=True)

# System packages required for CUDA extensions and TRELLIS dependencies
SYSTEM_PACKAGES = [
    "git",
    "ninja-build",
    "cmake",
    "build-essential",
    "clang",  # Required for CUDA extension builds
    "libgl1-mesa-glx",
    "libglib2.0-0",
    "libjpeg-dev",
    "libpng-dev",
    "libgomp1",  # OpenMP for parallel processing
]

# Core Python dependencies (from TRELLIS setup.sh --basic)
CORE_PYTHON_PACKAGES = [
    "pillow",
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "easydict",
    "opencv-python-headless",
    "scipy",
    "ninja",
    "rembg",
    "onnxruntime",
    "trimesh",
    "open3d",
    "xatlas",
    "pyvista",
    "pymeshfix",
    "igraph",
    "transformers",
    "huggingface-hub",
    "safetensors",
    "lz4",  # For state compression
]

# TRELLIS image with all CUDA extensions pre-compiled
trellis_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10",  # TRELLIS uses Python 3.10
    )
    .apt_install(*SYSTEM_PACKAGES)
    # Core Python dependencies (no GPU needed)
    .pip_install(*CORE_PYTHON_PACKAGES)
    # GPU build block - ALL CUDA-related builds in ONE block
    # because gpu="T4" creates a fresh build context
    .run_commands(
        # Install PyTorch with CUDA 11.8
        "pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118",
        # Build tools for --no-build-isolation
        "pip install wheel setuptools",
        # xformers for attention (pre-built wheel - use this instead of flash-attn)
        "pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118",
        # spconv (pre-built wheel)
        "pip install spconv-cu118",
        # nvdiffrast (builds from source, pinned for reproducibility)
        f"git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && cd /tmp/nvdiffrast && git checkout {PINNED_COMMITS['nvdiffrast']}",
        "pip install /tmp/nvdiffrast",
        # diffoctreerast (needs torch at build time, pinned)
        f"git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && cd /tmp/diffoctreerast && git checkout {PINNED_COMMITS['diffoctreerast']}",
        "pip install --no-build-isolation /tmp/diffoctreerast",
        # diff-gaussian-rasterization (needs torch at build time, pinned)
        f"git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && cd /tmp/mip-splatting && git checkout {PINNED_COMMITS['mip_splatting']}",
        "pip install --no-build-isolation /tmp/mip-splatting/submodules/diff-gaussian-rasterization/",
        # utils3d (pinned version for TRELLIS)
        f"pip install git+https://github.com/EasternJournalist/utils3d.git@{PINNED_COMMITS['utils3d']}",
        # kaolin for mesh processing (using cu121 wheel but compatible with cu118 runtime)
        "pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html",
        gpu="T4",
    )
    # Clone TRELLIS repository for imports (with submodules for flexicubes, pinned)
    .run_commands(
        f"git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /opt/TRELLIS && cd /opt/TRELLIS && git checkout {PINNED_COMMITS['trellis']}",
    )
    # Pre-download DINOv2 model to avoid 42s runtime download
    # TRELLIS uses dinov2_vitl14 with register tokens (dinov2_vitl14_reg4_pretrain.pth)
    .run_commands(
        "python -c \"import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', pretrained=True)\"",
    )
    # Set environment variables
    .env({
        "ATTN_BACKEND": "xformers",  # Use xformers for attention (flash-attn builds are too heavy)
        "SPCONV_ALGO": "native",  # Skip spconv runtime benchmarking
        "SPARSE_BACKEND": "spconv",  # Use spconv, not torchsparse
        "PYTHONPATH": "/opt/TRELLIS",  # Add TRELLIS to Python path
        "HF_HOME": HF_CACHE_PATH,  # HuggingFace cache in Modal volume
        "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9;9.0",  # Common GPU architectures
    })
)


@app.function(image=trellis_image, gpu="T4", timeout=300)
def verify_image():
    """
    Verify the image is correctly built with all dependencies.

    Run with: modal run trellis_modal/service/image.py::verify_image
    """
    import json

    results = {}

    # Test PyTorch + CUDA
    import torch
    results["pytorch_version"] = str(torch.__version__)
    results["cuda_version"] = str(torch.version.cuda)
    results["cuda_available"] = torch.cuda.is_available()
    results["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    # Test CUDA compute
    if torch.cuda.is_available():
        x = torch.randn(100, 100, device="cuda")
        _ = torch.matmul(x, x)  # Verify CUDA compute works
        results["cuda_compute_works"] = True

    # Test CUDA extensions
    import spconv.pytorch as spconv
    results["spconv"] = hasattr(spconv, 'SparseConvTensor')

    import nvdiffrast.torch as dr
    results["nvdiffrast"] = hasattr(dr, 'rasterize')

    import diffoctreerast
    results["diffoctreerast"] = hasattr(diffoctreerast, 'OctreeGaussianRasterizer')

    import diff_gaussian_rasterization
    results["gaussian_rasterization"] = hasattr(diff_gaussian_rasterization, 'GaussianRasterizer')

    import utils3d
    results["utils3d"] = hasattr(utils3d, '__version__') or True

    # Test DINOv2 is cached (should not download)
    import os
    dinov2_cache = os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth")
    dinov2_hub = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    results["dinov2_cached"] = os.path.exists(dinov2_cache) or os.path.exists(dinov2_hub)

    # Test TRELLIS imports
    try:
        from trellis import pipelines
        results["trellis_pipeline"] = hasattr(pipelines, 'TrellisImageTo3DPipeline')
    except ImportError as e:
        results["trellis_pipeline"] = str(e)

    try:
        from trellis import utils
        results["trellis_utils"] = hasattr(utils, 'render_utils')
    except ImportError as e:
        results["trellis_utils"] = str(e)

    print("=== Image Verification Results ===")
    for k, v in sorted(results.items()):
        status = "✓" if v is True else "✗" if v is False else "?"
        print(f"  {status} {k}: {v}")

    return json.dumps(results)


@app.local_entrypoint()
def main():
    """Run verification."""
    import json

    print("\n" + "="*60)
    print("TRELLIS Modal Image Verification")
    print("="*60 + "\n")

    result = verify_image.remote()
    data = json.loads(result)

    # Check critical components
    critical = [
        "cuda_available", "spconv", "nvdiffrast",
        "diffoctreerast", "gaussian_rasterization",
        "trellis_pipeline", "trellis_utils", "dinov2_cached"
    ]

    all_passed = all(data.get(k, False) is True for k in critical)

    if all_passed:
        print("\n✓ Image verification PASSED")
    else:
        print("\n✗ Image verification FAILED")
        for k in critical:
            v = data.get(k, "MISSING")
            status = "✓" if v is True else "✗"
            print(f"  {status} {k}: {v}")
