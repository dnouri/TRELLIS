# TRELLIS Modal Service Launch Checklist

Pre-launch verification checklist for production deployment.

## Pre-Deployment Checks

### Code Quality
- [ ] All tests passing: `pytest tests/ --ignore=tests/test_modal_e2e.py`
- [ ] No linting errors: `ruff check trellis_modal/service/ trellis_modal/client/ tests/`
- [ ] Type checks pass: `pyright trellis_modal/service/`

### Configuration
- [ ] `GPU_TYPE` set correctly in `config.py` (A100-40GB recommended)
- [ ] `MAX_IMAGE_PAYLOAD_SIZE` appropriate for use case (default: 10MB)
- [ ] `MAX_IMAGE_DIMENSION` appropriate for GPU memory (default: 4096)
- [ ] `CONTAINER_IDLE_TIMEOUT` balanced for cost vs latency (default: 300s)

### Security
- [ ] No hardcoded secrets in code
- [ ] API keys stored in Modal Volume (not in git)
- [ ] Production API keys have quotas set
- [ ] Test/dev keys excluded from production

## Deployment Steps

### 1. Environment Setup
- [ ] Modal CLI installed: `pip install modal`
- [ ] Modal authenticated: `modal token new`
- [ ] HuggingFace cache volume exists: `modal volume list`
- [ ] API keys volume exists: `modal volume list`

### 2. Initial Deployment
```bash
modal deploy -m trellis_modal.service.service
```
- [ ] Deployment completes without errors
- [ ] Container image builds successfully
- [ ] Model loads to CPU (check logs)
- [ ] GPU snapshot creates (if enabled)

### 3. Endpoint Verification

#### Health Check
```bash
curl https://<app>.modal.run/health
```
- [ ] Returns `{"status": "ok", "service": "trellis-api"}`

#### Generate Endpoint
```bash
curl -X POST https://<app>.modal.run/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_dev_xxx" \
  -d '{"image": "<base64>", "seed": 42}'
```
- [ ] Returns 200 with `state`, `video`, and `request_id`

#### Extract GLB Endpoint
- [ ] Returns 200 with `glb` and `request_id`

#### Extract Gaussian Endpoint
- [ ] Returns 200 with `ply` and `request_id`

### 4. Error Handling
- [ ] Missing API key returns 401 unauthorized
- [ ] Invalid API key returns 401 unauthorized
- [ ] Rate limited key returns 429 rate_limited
- [ ] Oversized image returns 400 validation_error
- [ ] Invalid image format returns 400 validation_error

## Post-Deployment Checks

### Monitoring
- [ ] Logs accessible in Modal dashboard
- [ ] Request IDs appearing in logs
- [ ] Error codes logged correctly

### Performance
- [ ] Cold start time acceptable (< 3 minutes)
- [ ] Warm request time acceptable (< 60 seconds)
- [ ] Memory usage stable (no leaks)

### Documentation
- [ ] OPERATIONS_RUNBOOK.md is current
- [ ] MODAL_INTEGRATION.md is current
- [ ] API documentation matches endpoints

## Go-Live Approval

| Role | Name | Date | Approved |
|------|------|------|----------|
| Developer | | | [ ] |
| Reviewer | | | [ ] |

## Rollback Plan

If issues occur:
1. `modal app stop trellis-service`
2. `git checkout <last-known-good-tag>`
3. `modal deploy -m trellis_modal.service.service`
4. Verify endpoints working
5. Investigate and fix issue
6. Redeploy when ready

## Notes

- First deployment takes longer (image build + model download)
- Subsequent deployments use cached image layers
- GPU snapshots require `modal deploy`, not `modal run`
- Keep at least one warm container for low latency
