# TRELLIS Modal Service Operations Runbook

Operational procedures for managing the TRELLIS 3D generation service on Modal.

## Quick Reference

| Action | Command |
|--------|---------|
| Deploy | `modal deploy -m trellis_modal.service.service` |
| Check logs | `modal app logs trellis-service` |
| List apps | `modal app list` |
| Add API key | `python -m trellis_modal.service.auth add-key --name <name>` |
| List keys | `python -m trellis_modal.service.auth list-keys` |

## Deployment

### Production Deployment

```bash
# Deploy with GPU memory snapshots enabled
modal deploy -m trellis_modal.service.service
```

This will:
1. Build the container image (cached after first build)
2. Load the TRELLIS model to CPU
3. Create a memory snapshot for faster cold starts
4. Expose HTTPS endpoints at `*.modal.run`

### Verify Deployment

```bash
# Check the health endpoint
curl https://<your-modal-app>.modal.run/health

# Expected response:
# {"status": "ok", "service": "trellis-api"}
```

### Rollback

```bash
# List deployed apps
modal app list

# Stop current deployment
modal app stop trellis-service

# Redeploy previous version (checkout git tag first)
git checkout <previous-tag>
modal deploy -m trellis_modal.service.service
```

## API Key Management

API keys are stored in a Modal Volume (`trellis-api-keys`) at `/data/keys.json`.

### Add a New Key

```bash
# Development key (no quota)
python -m trellis_modal.service.auth add-key --name "dev-user"

# Production key with quota
python -m trellis_modal.service.auth add-key --name "customer-xyz" --prefix live --quota 1000
```

### List All Keys

```bash
python -m trellis_modal.service.auth list-keys
```

### Revoke a Key

```bash
python -m trellis_modal.service.auth revoke-key sk_dev_xxxxx
```

### Check Key Usage

```bash
python -m trellis_modal.service.auth usage sk_dev_xxxxx
```

## Monitoring

### Log Analysis

View service logs in Modal dashboard or CLI:

```bash
modal app logs trellis-service --since 1h
```

Key log patterns to monitor:

| Pattern | Meaning |
|---------|---------|
| `Request completed:.*status.*success` | Successful request |
| `Request failed:.*error_code.*cuda_oom` | GPU memory exhaustion |
| `Auth failed` | Invalid/expired API key |
| `Rate limit exceeded` | Key hit quota |

### Metrics

Metrics are logged in structured format. Key fields:

- `request_id`: Unique ID for tracing
- `endpoint`: generate, extract_glb, extract_gaussian
- `duration_ms`: Request latency
- `status`: success or error
- `output_size_bytes`: Response payload size

## Incident Response

### GPU Out of Memory (cuda_oom)

**Symptoms:** Requests failing with `cuda_oom` error code.

**Response:**
1. Check if a specific image is causing issues (large dimensions)
2. Verify input validation limits are enforced (max 4096x4096)
3. If persistent, may need to restart container or reduce concurrent requests

### High Error Rate

**Symptoms:** Many requests failing.

**Response:**
1. Check Modal dashboard for container health
2. Review recent logs for error patterns
3. Verify model is loaded (`/health` endpoint returns healthy)
4. Check if issue is specific to certain API keys

### Cold Start Latency

**Symptoms:** First request after idle takes 2-3 minutes.

**Response:**
1. This is expected behavior when container scales from zero
2. Consider increasing `scaledown_window` in config (default: 300s)
3. GPU memory snapshots reduce this to ~1-2 seconds

## Scaling

### Current Limits

| Resource | Limit |
|----------|-------|
| GPU | A100-40GB |
| Max payload | 10 MB |
| Max dimensions | 4096x4096 |
| Container idle | 300 seconds |

### Adjusting Concurrency

Edit `trellis_modal/service/image.py` and modify the `@app.cls` decorator:

```python
@app.cls(
    gpu=gpu.A100(size="40GB"),
    concurrency_limit=2,  # Max concurrent requests per container
    allow_concurrent_inputs=2,
)
```

### Adjusting Idle Timeout

In `trellis_modal/service/config.py`:

```python
CONTAINER_IDLE_TIMEOUT = 300  # Increase to keep containers warm longer
```

## Troubleshooting

### Container Won't Start

1. Check Modal dashboard for build errors
2. Verify HuggingFace cache volume exists
3. Check if model download is stuck (network issues)

### Requests Timing Out

1. Check if model is loaded (cold start in progress?)
2. Verify GPU is available (`health_check()` in logs)
3. Check for memory pressure (large batches)

### API Key Not Working

1. Verify key format starts with `sk_`
2. Check if key is active (`list-keys`)
3. Check if rate limit exceeded (`usage`)
4. Verify keys.json is in the volume

## Maintenance

### Updating the Model

1. Update `MODEL_NAME` in `config.py`
2. Clear HuggingFace cache if needed
3. Redeploy: `modal deploy -m trellis_modal.service.service`

### Backup API Keys

```bash
# Download keys from volume
modal volume get trellis-api-keys keys.json ./backup/keys.json
```

### Rotate Secrets

1. Generate new API keys for affected users
2. Revoke old keys
3. Notify users of new keys
