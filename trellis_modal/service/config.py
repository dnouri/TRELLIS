"""
Configuration constants for the Modal TRELLIS service.

Contains deployment settings, resource limits, and path configurations
that are shared across the service modules.
"""

# Modal resource configuration
GPU_TYPE = "A100-40GB"
GPU_MEMORY_SNAPSHOT = True  # Experimental feature for faster cold starts
CONTAINER_IDLE_TIMEOUT = 300  # seconds

# Volume mount paths
HF_CACHE_PATH = "/cache/huggingface"
API_KEYS_PATH = "/data/keys.json"

# Model configuration
MODEL_NAME = "JeffreyXiang/TRELLIS-image-large"

# Input validation limits
MAX_IMAGE_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10MB (decoded binary size)
MAX_IMAGE_DIMENSION = 4096  # Max width or height in pixels

# Generation defaults
DEFAULT_SEED = 42
DEFAULT_SLAT_SAMPLER_PARAMS = {
    "steps": 12,
    "cfg_strength": 7.5,
}
DEFAULT_SS_SAMPLING_STEPS = 12
DEFAULT_MESH_SIMPLIFY_RATIO = 0.95
DEFAULT_TEXTURE_SIZE = 1024
