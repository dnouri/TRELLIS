# TRELLIS Modal Integration

GPU-accelerated 3D generation from images via Modal serverless infrastructure.

## Architecture Overview

```
+-------------------------------------------------------------+
|  LOCAL GRADIO CLIENT                                        |
|  - Runs on user's machine                                   |
|  - Handles image upload, parameter controls, result display |
|  - Stores compressed state in gr.State                      |
|  - Communicates via HTTPS                                   |
+-----------------------------+-------------------------------+
                              |
                              | POST /generate, POST /extract_glb
                              | Headers: X-API-Key: sk_xxx
                              v
+-------------------------------------------------------------+
|  MODAL WEB ENDPOINTS                                        |
|                                                             |
|  Authentication Layer (API Key validation from Volume)      |
|                           |                                 |
|                           v                                 |
|  +-------------------------------------------------------+  |
|  |  TRELLISService Class                                 |  |
|  |  gpu="A100-40GB"                                      |  |
|  |  enable_memory_snapshot=True                          |  |
|  |                                                       |  |
|  |  @enter(snap=True): Load 7 models to CPU              |  |
|  |  @enter(snap=False): Move models to GPU               |  |
|  |  generate(): Image -> 3D state + video                |  |
|  |  extract_glb(): State -> GLB mesh                     |  |
|  |  extract_gaussian(): State -> PLY point cloud         |  |
|  +-------------------------------------------------------+  |
|                                                             |
|  Volumes:                                                   |
|  - /cache/huggingface: Model weights cache                  |
|  - /data/keys.json: API key storage                         |
+-------------------------------------------------------------+
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CUDA Extensions | Baked into Modal image | Reproducibility, zero runtime compilation |
| Cold Start Mitigation | GPU Memory Snapshots | ~150s -> seconds on restore |
| State Transfer | LZ4 compression, client storage | Stateless server, ~75% size reduction |
| GPU | A100-40GB | Cost-effective, 23GB peak VRAM fits |
| File Delivery | Base64 in JSON response | Simple, no external storage needed |
| Authentication | API keys in Modal Volume | Multi-key support without external DB |

---

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: `pip install modal`
3. **Modal Authentication**: `modal token new`
4. **Python 3.10+** for client

---

## Quick Start

### 1. Deploy the Service

```bash
# Clone repository
git clone https://github.com/your-org/TRELLIS.git
cd TRELLIS

# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new

# Create Modal volumes
modal volume create trellis-hf-cache
modal volume create trellis-api-keys

# Upload initial API keys
python -m trellis_modal.service.auth add-key --name "development"
# Note the generated key (sk_dev_xxx...)

# Deploy (first deployment builds image, takes ~45 minutes)
modal deploy -m trellis_modal.service.service
```

### 2. Run the Client

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install client dependencies
pip install -r trellis_modal/client/requirements.txt

# Set environment variables
export TRELLIS_API_URL=https://your-modal-url.modal.run
export TRELLIS_API_KEY=sk_dev_xxxxx

# Run client
python -m client.app
```

### 3. Generate 3D

1. Open http://localhost:7860 in your browser
2. Upload an image
3. Adjust parameters (optional)
4. Click "Generate"
5. Wait for preview video (~60-90s)
6. Click "Extract GLB" for mesh file

---

## API Reference

### Authentication

All endpoints require an `X-API-Key` header:

```bash
curl -H "X-API-Key: sk_dev_xxxxx" ...
```

### POST /generate

Generate 3D model from image.

**Request:**
```json
{
  "image": "<base64-encoded-image>",
  "seed": 42,
  "ss_sampling_steps": 12,
  "slat_sampling_steps": 12,
  "slat_guidance_strength": 3.0
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | required | Base64-encoded PNG/JPG |
| `seed` | int | 42 | Random seed for reproducibility |
| `ss_sampling_steps` | int | 12 | Sparse structure sampling steps |
| `slat_sampling_steps` | int | 12 | SLAT sampling steps |
| `slat_guidance_strength` | float | 3.0 | Classifier-free guidance |

**Response (success):**
```json
{
  "state": "<base64-compressed-state>",
  "video": "<base64-mp4-preview>"
}
```

**Response (error):**
```json
{
  "error": {
    "code": "cuda_oom",
    "message": "GPU out of memory. Try a smaller image."
  }
}
```

### POST /extract_glb

Extract GLB mesh from generation state.

**Request:**
```json
{
  "state": "<base64-compressed-state-from-generate>",
  "mesh_simplify_ratio": 0.95,
  "texture_size": 1024
}
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state` | string | required | State from /generate |
| `mesh_simplify_ratio` | float | 0.95 | Mesh simplification (0-1) |
| `texture_size` | int | 1024 | Texture resolution (512/1024/2048) |

**Response:**
```json
{
  "glb": "<base64-glb-file>"
}
```

### POST /extract_gaussian

Extract Gaussian PLY from generation state.

**Request:**
```json
{
  "state": "<base64-compressed-state-from-generate>"
}
```

**Response:**
```json
{
  "ply": "<base64-ply-file>"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `unauthorized` | 401 | Invalid or missing API key |
| `validation_error` | 400 | Invalid request format or parameters |
| `cuda_oom` | 500 | GPU out of memory |
| `generation_error` | 500 | Pipeline failure |
| `extraction_error` | 500 | GLB/PLY extraction failed |

---

## Configuration

### Environment Variables (Client)

| Variable | Required | Description |
|----------|----------|-------------|
| `TRELLIS_API_URL` | Yes | Modal endpoint base URL |
| `TRELLIS_API_KEY` | Yes | API key for authentication |

### Modal Configuration

Configuration constants in `trellis_modal/service/config.py`:

```python
GPU_TYPE = "A100-40GB"           # GPU type
GPU_MEMORY_SNAPSHOT = True       # Enable GPU snapshots
CONTAINER_IDLE_TIMEOUT = 300     # Keep container warm for 5 min
MODEL_NAME = "JeffreyXiang/TRELLIS-image-large"  # HuggingFace model
```

### API Key Management

```bash
# Add new key
python -m trellis_modal.service.auth add-key --name "user1"

# List keys
python -m trellis_modal.service.auth list-keys

# Revoke key
python -m trellis_modal.service.auth revoke-key sk_dev_xxxxx
```

---

## Deployment

### Initial Deployment

```bash
# Deploy to Modal (builds image on first run)
modal deploy -m trellis_modal.service.service
```

First deployment takes ~45 minutes to build the Docker image with CUDA extensions.

### Updating

```bash
# Redeploy with code changes
modal deploy -m trellis_modal.service.service
```

Code-only changes deploy in ~30 seconds. Image changes trigger rebuild.

### Monitoring

```bash
# View logs
modal logs

# Check app status
modal app list

# View volume contents
modal volume ls trellis-api-keys
```

### Rollback

Modal keeps previous deployments. To rollback:

```bash
# List deployments
modal app list

# Rollback (stop current, previous becomes active)
modal app stop <app-id>
```

---

## Performance

### Benchmark Results (A100-40GB)

| Metric | Value |
|--------|-------|
| Cold start (no snapshot) | ~150s |
| Cold start (with snapshot) | ~5-10s |
| Generate (warm) | ~60-90s |
| Extract GLB | ~45-60s |
| Extract Gaussian | ~1s |
| VRAM usage | 5.73 GB idle, ~23 GB peak |

### Cost Estimation

- A100-40GB: $2.10/hour on Modal
- Generation (~90s): ~$0.05
- GLB extraction (~60s): ~$0.035
- **Total per model**: ~$0.09

### Optimization Tips

1. **Keep containers warm**: Default 5-minute idle timeout
2. **Use GPU snapshots**: Reduces cold start from 150s to ~5s
3. **Batch requests**: Process multiple images per warm container
4. **Reduce sampling steps**: Faster but lower quality

---

## Troubleshooting

### Connection Refused

```
requests.exceptions.ConnectionError: Connection refused
```

**Cause**: Service not deployed or container not started.

**Fix**: Deploy service: `modal deploy -m trellis_modal.service.service`

### 401 Unauthorized

```json
{"error": {"code": "unauthorized", "message": "Invalid or missing API key"}}
```

**Cause**: Invalid API key or missing header.

**Fix**:
1. Verify `TRELLIS_API_KEY` environment variable
2. Generate new key: `python -m trellis_modal.service.auth add-key --name "test"`

### CUDA Out of Memory

```json
{"error": {"code": "cuda_oom", "message": "GPU out of memory"}}
```

**Cause**: Input image too large or concurrent requests.

**Fix**:
1. Resize input image to 512x512
2. Reduce `texture_size` parameter
3. Wait for other requests to complete

### Slow Cold Start

**Cause**: GPU memory snapshots not active.

**Note**: Snapshots only work with `modal deploy`, not `modal run`.

**Fix**: Deploy service: `modal deploy -m trellis_modal.service.service`

### Image Build Fails

**Cause**: CUDA extension compilation error.

**Fix**:
1. Check Modal logs: `modal logs`
2. Verify base image compatibility
3. File issue with error details

---

## Development

### Local Testing

```bash
# Run tests (no GPU needed)
pytest tests/

# Run specific test
pytest tests/test_compression.py -v

# Run integration tests (requires deployed service)
pytest tests/ -m integration
```

### Running Service Locally (for debugging)

```bash
# Run with Modal's local simulator
modal run -m trellis_modal.service.service::test_service
```

Note: GPU snapshots don't work with `modal run`, only `modal deploy`.

### Project Structure

```
trellis_modal/service/
  __init__.py           # Package init
  config.py             # Configuration constants
  image.py              # Modal image definition
  generator.py          # TRELLISGenerator class
  service.py            # Modal service with endpoints
  auth.py               # API key management
  streaming.py          # SSE utilities
  state.py              # State pack/unpack

trellis_modal/client/
  __init__.py
  app.py                # Gradio interface
  api.py                # API client
  compression.py        # LZ4 compression
  requirements.txt      # Client dependencies

tests/
  conftest.py           # Test fixtures
  test_*.py             # Test modules

scripts/
  benchmark.py          # Performance benchmarking
```

---

## Security

### API Key Security

- Keys are stored in Modal Volume (encrypted at rest)
- Keys are masked in logs (`sk_dev_xxx...xxx`)
- Use `sk_dev_` prefix for development, `sk_live_` for production

### State Security

- State uses pickle serialization
- Only decompress state from trusted sources
- State contains model outputs, not user data

### Network Security

- All traffic over HTTPS
- Modal handles TLS termination
- No data stored on server between requests

---

## Cost Management

### Monitor Usage

```bash
# View Modal usage
modal usage

# View API key usage
python -m trellis_modal.service.auth usage sk_dev_xxxxx
```

### Reduce Costs

1. **Shorter idle timeout**: Reduce `CONTAINER_IDLE_TIMEOUT` in config
2. **Smaller GPU**: Use T4 for testing (not recommended for production)
3. **Rate limiting**: Implement per-key quotas

### Budget Alerts

Configure in Modal dashboard: https://modal.com/billing

---

## Support

- **Issues**: https://github.com/your-org/TRELLIS/issues
- **Modal Docs**: https://modal.com/docs
- **TRELLIS Paper**: [arXiv link]
