# Design Notes

This document explains why we built what we built.

## The Problem

TRELLIS is a 3D generation model. It runs on GPUs. Most people don't have GPUs that can run it. Those who do must install CUDA extensions, which is tedious and error-prone.

We wanted to let anyone generate 3D models from images without owning specialized hardware or compiling anything.

## Why Modal

Serverless GPU compute solves both problems at once. You upload code, they run it on GPUs, you pay per second. No servers to manage. No hardware to buy.

Modal specifically offers GPU memory snapshots—an experimental feature that captures the GPU state after model loading. On subsequent requests, instead of reloading 7 models (~150 seconds), the snapshot restores in seconds. This matters because cold starts are the primary cost driver for occasional use.

We considered alternatives. Running a dedicated GPU instance costs ~$1,500/month whether you use it or not. Modal costs ~$0.09 per generation, only when you generate.

## Architecture Decisions

**Baked CUDA extensions.** TRELLIS depends on several custom CUDA kernels (nvdiffrast, diffoctreerast, spconv, diff-gaussian-rasterization). These must be compiled against specific PyTorch and CUDA versions. We compile them once into the Modal image. Users never touch a compiler.

**Client-side state.** After generation, the server has ~50MB of intermediate state (Gaussians, mesh) needed for export. We compress it with LZ4 (~75% reduction), send it to the client, and let the client send it back for export. The server stays stateless. Stateless servers are simpler to scale and reason about.

**API keys over OAuth.** We store keys in a Modal Volume as JSON. No database, no external auth service. Simple systems have fewer failure modes. Keys are prefixed (`sk_dev_`, `sk_live_`) so you know what you're looking at.

**Base64 over presigned URLs.** GLB files are 10-50MB. We could store them temporarily and return a URL. Instead, we base64-encode and return directly. One fewer moving part. One fewer thing to expire or clean up.

**A100-40GB.** Peak VRAM during generation is ~23GB. The A100-40GB fits this with headroom and costs $2.10/hour. The A10G (24GB, $1.10/hour) would be tight. We chose reliability over cost savings.

## What Success Looks Like

1. **It works.** Image goes in, 3D model comes out.
2. **It's fast enough.** Under 90 seconds for generation, under 60 for export.
3. **It's cheap enough.** Under $0.10 per model.
4. **It's simple.** Three commands to deploy, two environment variables to configure, one Gradio interface to use.

## What We Built

```
trellis_modal/service/    Server code that runs on Modal
  image.py        Container image with CUDA extensions
  service.py      HTTP endpoints and request handling
  generator.py    TRELLIS pipeline wrapper
  auth.py         API key validation
  state.py        State serialization
  config.py       Constants

trellis_modal/client/           Local code that runs on your machine
  app.py          Gradio interface
  api.py          HTTP client
  compression.py  LZ4 state compression
```

The server exposes three endpoints: `/generate` (image → state + preview), `/extract_glb` (state → mesh), `/extract_gaussian` (state → point cloud). The client calls them.

## SSE Event Format

Generation progress uses Server-Sent Events. The format:

```
event: progress
data: {"stage": "sparse_structure", "progress": 0.5, "message": "Generating structure..."}

event: complete
data: {"state": "base64...", "video": "base64..."}

event: error
data: {"code": "cuda_oom", "message": "GPU out of memory"}
```

Stages: `preprocessing`, `encoding`, `sparse_structure`, `structured_latent`, `decoding`, `rendering`.

Error codes: `unauthorized`, `validation_error`, `rate_limited`, `cuda_oom`, `generation_error`, `extraction_error`.

## What We Didn't Build

**Streaming GLB delivery.** GLBs come back as one blob. For very large meshes, chunked streaming would reduce memory pressure. We didn't need it yet.

**Multi-region deployment.** Modal can deploy to specific regions. We deploy to their default. Latency matters less when generation takes 90 seconds.

**Automatic scaling policies.** Modal handles scaling. We set idle timeout (5 minutes) and let it decide when to spin down. If traffic patterns emerge that justify tuning, we'll tune.

**Usage-based billing passthrough.** We track API key usage but don't bill for it. Keys have optional quotas. Monetization is left as an exercise.

## Reading Order

1. This document (you are here)
2. [MODAL_INTEGRATION.md](MODAL_INTEGRATION.md) — How to deploy and use it
3. [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) — How to operate it
4. [LAUNCH_CHECKLIST.md](LAUNCH_CHECKLIST.md) — What to verify before launch
