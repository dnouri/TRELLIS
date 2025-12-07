"""
Test that ensures constants are consistent across modules.

Modal's execution model copies image.py to /root/image.py without the
modal_service package context, so imports from config.py fail. We must
duplicate constants but verify they stay in sync with this test.
"""

from trellis_modal.service import config
from trellis_modal.service import image


def test_gpu_type_matches():
    """GPU_TYPE must match between config and image modules."""
    assert image.GPU_TYPE == config.GPU_TYPE, (
        f"GPU_TYPE mismatch: image.py has '{image.GPU_TYPE}' "
        f"but config.py has '{config.GPU_TYPE}'"
    )


def test_hf_cache_path_matches():
    """HF_CACHE_PATH must match between config and image modules."""
    assert image.HF_CACHE_PATH == config.HF_CACHE_PATH, (
        f"HF_CACHE_PATH mismatch: image.py has '{image.HF_CACHE_PATH}' "
        f"but config.py has '{config.HF_CACHE_PATH}'"
    )


def test_model_name_matches():
    """MODEL_NAME must match between config and image modules."""
    assert image.MODEL_NAME == config.MODEL_NAME, (
        f"MODEL_NAME mismatch: image.py has '{image.MODEL_NAME}' "
        f"but config.py has '{config.MODEL_NAME}'"
    )
