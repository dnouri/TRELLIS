#!/usr/bin/env python3
"""
Performance benchmark script for TRELLIS Modal service.

Measures end-to-end timing for generation and extraction operations.
Outputs JSON metrics suitable for cost analysis and performance tracking.

Usage:
    export TRELLIS_API_URL=https://your-modal-url.modal.run
    export TRELLIS_API_KEY=sk_dev_xxxxx
    python scripts/benchmark.py [--iterations N] [--image PATH]

Requirements:
    - Deployed Modal service with GPU
    - Valid API key
    - Test image file
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import requests


def encode_image(image_path: str) -> str:
    """Encode image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def benchmark_generate(
    base_url: str,
    api_key: str,
    image_b64: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Benchmark a single generate request.

    Returns:
        Dict with timing metrics and response info
    """
    start = time.perf_counter()

    response = requests.post(
        f"{base_url}/generate",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={
            "image": image_b64,
            "seed": params.get("seed", 42),
            "ss_sampling_steps": params.get("ss_sampling_steps", 12),
            "slat_sampling_steps": params.get("slat_sampling_steps", 12),
            "slat_guidance_strength": params.get("slat_guidance_strength", 3.0),
        },
        timeout=600,  # 10 min timeout for cold starts
    )

    elapsed = time.perf_counter() - start

    result = response.json()

    if "error" in result:
        return {
            "success": False,
            "elapsed_seconds": elapsed,
            "error": result["error"],
        }

    return {
        "success": True,
        "elapsed_seconds": elapsed,
        "state_size_bytes": len(result.get("state", "")),
        "video_size_bytes": len(result.get("video", "")),
    }


def benchmark_extract_glb(
    base_url: str,
    api_key: str,
    state: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Benchmark a single extract_glb request.

    Returns:
        Dict with timing metrics and response info
    """
    start = time.perf_counter()

    response = requests.post(
        f"{base_url}/extract_glb",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={
            "state": state,
            "mesh_simplify_ratio": params.get("mesh_simplify_ratio", 0.95),
            "texture_size": params.get("texture_size", 1024),
        },
        timeout=600,
    )

    elapsed = time.perf_counter() - start

    result = response.json()

    if "error" in result:
        return {
            "success": False,
            "elapsed_seconds": elapsed,
            "error": result["error"],
        }

    return {
        "success": True,
        "elapsed_seconds": elapsed,
        "glb_size_bytes": len(base64.b64decode(result.get("glb", ""))),
    }


def run_benchmarks(
    base_url: str,
    api_key: str,
    image_path: str,
    iterations: int = 3,
) -> dict[str, Any]:
    """
    Run complete benchmark suite.

    Args:
        base_url: Modal endpoint base URL
        api_key: API key for authentication
        image_path: Path to test image
        iterations: Number of iterations for warm benchmarks

    Returns:
        Complete benchmark results
    """
    print("TRELLIS Performance Benchmark")
    print(f"{'=' * 50}")
    print(f"Endpoint: {base_url}")
    print(f"Iterations: {iterations}")
    print(f"Image: {image_path}")
    print()

    # Encode image
    image_b64 = encode_image(image_path)
    print(f"Image size: {len(image_b64)} bytes (base64)")
    print()

    results: dict[str, Any] = {
        "endpoint": base_url,
        "image_path": image_path,
        "iterations": iterations,
        "generate": {"runs": []},
        "extract_glb": {"runs": []},
    }

    # Cold start test (first request)
    print("Running cold start test (generate)...")
    cold_result = benchmark_generate(base_url, api_key, image_b64, {})
    results["generate"]["cold_start"] = cold_result
    print(f"  Cold start: {cold_result['elapsed_seconds']:.2f}s")

    if not cold_result["success"]:
        print(f"  ERROR: {cold_result['error']}")
        return results

    state = None
    # Warm runs
    print(f"\nRunning {iterations} warm iterations (generate)...")
    for i in range(iterations):
        result = benchmark_generate(base_url, api_key, image_b64, {"seed": 42 + i})
        results["generate"]["runs"].append(result)
        print(f"  Run {i + 1}: {result['elapsed_seconds']:.2f}s")
        if result["success"]:
            state = result.get("state")

    # Calculate generate statistics
    successful_runs = [r for r in results["generate"]["runs"] if r["success"]]
    if successful_runs:
        times = [r["elapsed_seconds"] for r in successful_runs]
        results["generate"]["stats"] = {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
        }
        print(f"\n  Mean: {results['generate']['stats']['mean']:.2f}s")
        print(f"  Stdev: {results['generate']['stats']['stdev']:.2f}s")

    # Extract GLB benchmarks
    if state:
        print(f"\nRunning {iterations} extract_glb iterations...")
        for i in range(iterations):
            result = benchmark_extract_glb(base_url, api_key, state, {})
            results["extract_glb"]["runs"].append(result)
            print(f"  Run {i + 1}: {result['elapsed_seconds']:.2f}s")

        # Calculate extract statistics
        successful_runs = [r for r in results["extract_glb"]["runs"] if r["success"]]
        if successful_runs:
            times = [r["elapsed_seconds"] for r in successful_runs]
            results["extract_glb"]["stats"] = {
                "mean": statistics.mean(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times),
            }
            print(f"\n  Mean: {results['extract_glb']['stats']['mean']:.2f}s")
            print(f"  Stdev: {results['extract_glb']['stats']['stdev']:.2f}s")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="TRELLIS Modal performance benchmark")
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=3,
        help="Number of warm iterations (default: 3)",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default="assets/example_image/typical_building_building.png",
        help="Path to test image",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: stdout)",
    )

    args = parser.parse_args()

    # Get configuration from environment
    base_url = os.environ.get("TRELLIS_API_URL")
    api_key = os.environ.get("TRELLIS_API_KEY")

    if not base_url or not api_key:
        print("Error: TRELLIS_API_URL and TRELLIS_API_KEY environment variables required")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Run benchmarks
    results = run_benchmarks(
        base_url=base_url,
        api_key=api_key,
        image_path=args.image,
        iterations=args.iterations,
    )

    # Output results
    print("\n" + "=" * 50)
    print("Benchmark Complete")
    print("=" * 50)

    json_output = json.dumps(results, indent=2)

    if args.output:
        Path(args.output).write_text(json_output)
        print(f"Results saved to: {args.output}")
    else:
        print("\nJSON Results:")
        print(json_output)


if __name__ == "__main__":
    main()
