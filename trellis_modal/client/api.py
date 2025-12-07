"""
API client for communicating with the Modal TRELLIS service.

Handles HTTP requests to Modal endpoints with:
- API key authentication
- Error handling and response parsing
- Timeout handling for cold starts
- Cold start detection
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable

import requests


class TrellisAPIClient:
    """
    Client for the Modal-deployed TRELLIS service.

    Handles all communication with the server including authentication,
    request formatting, and response parsing.
    """

    # Default timeout for requests (5 minutes to handle cold starts)
    DEFAULT_TIMEOUT = 300

    # Default cold start threshold in seconds
    DEFAULT_COLD_START_THRESHOLD = 5.0

    # Retry configuration
    MAX_RETRIES = 2  # Number of retries after initial attempt
    INITIAL_BACKOFF = 1.0  # Initial backoff in seconds

    def __init__(self, base_url: str, api_key: str) -> None:
        """
        Initialize the API client.

        Args:
            base_url: Modal endpoint base URL
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._last_request_elapsed: float | None = None

    @property
    def last_request_elapsed(self) -> float | None:
        """Return elapsed time of last request in seconds, or None if no request made."""
        return self._last_request_elapsed

    def was_cold_start(self, threshold: float | None = None) -> bool:
        """
        Check if the last request was likely a cold start.

        Args:
            threshold: Time threshold in seconds (default: 5.0)

        Returns:
            True if last request took longer than threshold, False otherwise
        """
        if self._last_request_elapsed is None:
            return False
        if threshold is None:
            threshold = self.DEFAULT_COLD_START_THRESHOLD
        return self._last_request_elapsed > threshold

    def _headers(self) -> dict[str, str]:
        """Return headers for API requests."""
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """
        Make HTTP request with retry logic for transient failures.

        Retries on ConnectionError and Timeout with exponential backoff.
        Does not retry on successful responses (even if they contain API errors).

        Args:
            method: HTTP method (e.g., "POST")
            url: Request URL
            **kwargs: Additional arguments for requests

        Returns:
            Response object

        Raises:
            requests.exceptions.ConnectionError: After exhausting retries
            requests.exceptions.Timeout: After exhausting retries
        """
        last_exception = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return requests.request(method, url, **kwargs)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_exception = e
                if attempt < self.MAX_RETRIES:
                    # Exponential backoff: 1s, 2s, 4s, etc.
                    delay = self.INITIAL_BACKOFF * (2 ** attempt)
                    time.sleep(delay)

        # All retries exhausted
        raise last_exception  # type: ignore[misc]

    def _check_error(self, response_data: dict[str, Any]) -> None:
        """
        Check response for errors and raise APIError if found.

        Args:
            response_data: Parsed JSON response

        Raises:
            APIError: If response contains an error
        """
        if "error" in response_data:
            error = response_data["error"]
            raise APIError(error["code"], error["message"])

    def generate(
        self,
        image_path: str,
        seed: int,
        ss_sampling_steps: int,
        slat_sampling_steps: int,
        slat_guidance_strength: float,
        on_progress: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Generate 3D from image via the Modal service.

        Args:
            image_path: Path to input image
            seed: Random seed for reproducibility
            ss_sampling_steps: Sparse structure sampling steps
            slat_sampling_steps: SLAT sampling steps
            slat_guidance_strength: Guidance strength
            on_progress: Optional callback for progress updates (not used for sync)

        Returns:
            Dict with 'state' (base64 compressed state) and 'video' (base64)

        Raises:
            APIError: If the request fails
            FileNotFoundError: If image_path doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read and encode image
        image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")

        payload = {
            "image": image_b64,
            "seed": seed,
            "ss_sampling_steps": ss_sampling_steps,
            "slat_sampling_steps": slat_sampling_steps,
            "slat_guidance_strength": slat_guidance_strength,
        }

        start = time.perf_counter()
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/generate",
            headers=self._headers(),
            json=payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        self._last_request_elapsed = time.perf_counter() - start

        result = response.json()
        self._check_error(result)

        return result

    def extract_glb(
        self,
        state: str,
        mesh_simplify_ratio: float,
        texture_size: int,
        output_path: str,
        on_progress: Callable[[str, float, str], None] | None = None,
    ) -> str:
        """
        Extract GLB mesh from generation state.

        Args:
            state: Base64 compressed state string from generate()
            mesh_simplify_ratio: Mesh simplification ratio
            texture_size: Texture resolution
            output_path: Path to write GLB file
            on_progress: Optional callback for progress updates (not used for sync)

        Returns:
            Path to written GLB file

        Raises:
            APIError: If the request fails
        """
        payload = {
            "state": state,
            "mesh_simplify_ratio": mesh_simplify_ratio,
            "texture_size": texture_size,
        }

        start = time.perf_counter()
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/extract_glb",
            headers=self._headers(),
            json=payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        self._last_request_elapsed = time.perf_counter() - start

        result = response.json()
        self._check_error(result)

        # Decode and write GLB
        glb_bytes = base64.b64decode(result["glb"])
        Path(output_path).write_bytes(glb_bytes)

        return output_path

    def extract_gaussian(
        self,
        state: str,
        output_path: str,
        on_progress: Callable[[str, float, str], None] | None = None,
    ) -> str:
        """
        Extract Gaussian splat from generation state.

        Args:
            state: Base64 compressed state string from generate()
            output_path: Path to write PLY file
            on_progress: Optional callback for progress updates (not used for sync)

        Returns:
            Path to written PLY file

        Raises:
            APIError: If the request fails
        """
        payload = {
            "state": state,
        }

        start = time.perf_counter()
        response = self._request_with_retry(
            "POST",
            f"{self.base_url}/extract_gaussian",
            headers=self._headers(),
            json=payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        self._last_request_elapsed = time.perf_counter() - start

        result = response.json()
        self._check_error(result)

        # Decode and write PLY
        ply_bytes = base64.b64decode(result["ply"])
        Path(output_path).write_bytes(ply_bytes)

        return output_path

    def health_check(self) -> bool:
        """
        Check if the service is reachable.

        Makes a minimal HEAD request to the base URL. Returns True if any HTTP
        response is received (even error codes), False if the connection fails.

        Returns:
            True if service is reachable, False otherwise
        """
        try:
            requests.request(
                "HEAD",
                self.base_url,
                headers=self._headers(),
                timeout=10,
            )
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False


class APIError(Exception):
    """Exception raised for API errors."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")
