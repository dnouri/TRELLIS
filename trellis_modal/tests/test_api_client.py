"""
Tests for the API client.

Tests validate TrellisAPIClient HTTP calls, error handling, and response parsing.
"""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from trellis_modal.client.api import APIError, TrellisAPIClient


class TestTrellisAPIClientInit:
    """Tests for TrellisAPIClient initialization."""

    def test_init_stores_base_url_and_api_key(self) -> None:
        """Client should store base_url and api_key."""
        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )
        assert client.base_url == "https://example.com"
        assert client.api_key == "sk_dev_test123"

    def test_init_strips_trailing_slash_from_base_url(self) -> None:
        """Base URL trailing slash should be stripped."""
        client = TrellisAPIClient(
            base_url="https://example.com/",
            api_key="sk_dev_test123",
        )
        assert client.base_url == "https://example.com"


class TestTrellisAPIClientGenerate:
    """Tests for TrellisAPIClient.generate()."""

    def test_generate_sends_correct_request(self, tmp_path: Path) -> None:
        """generate() should send POST with correct headers and payload."""
        # Create test image
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            client.generate(
                image_path=str(image_path),
                seed=42,
                ss_sampling_steps=12,
                slat_sampling_steps=12,
                slat_guidance_strength=3.0,
            )

            # Verify call
            mock_request.assert_called_once()
            call_args = mock_request.call_args

            # Check method and URL
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "https://example.com/generate"

            # Check headers
            assert call_args[1]["headers"]["X-API-Key"] == "sk_dev_test123"
            assert call_args[1]["headers"]["Content-Type"] == "application/json"

            # Check payload
            payload = call_args[1]["json"]
            assert payload["seed"] == 42
            assert payload["ss_sampling_steps"] == 12
            assert payload["slat_sampling_steps"] == 12
            assert payload["slat_guidance_strength"] == 3.0
            # Image should be base64-encoded
            assert payload["image"] == base64.b64encode(b"fake png data").decode()

    def test_generate_returns_result_dict(self, tmp_path: Path) -> None:
        """generate() should return dict with state and video."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            result = client.generate(
                image_path=str(image_path),
                seed=42,
                ss_sampling_steps=12,
                slat_sampling_steps=12,
                slat_guidance_strength=3.0,
            )

            assert "state" in result
            assert "video" in result

    def test_generate_raises_api_error_on_error_response(self, tmp_path: Path) -> None:
        """generate() should raise APIError on error response."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_bad",
        )

        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"error": {"code": "unauthorized", "message": "Invalid key"}},
            )

            with pytest.raises(APIError) as exc_info:
                client.generate(
                    image_path=str(image_path),
                    seed=42,
                    ss_sampling_steps=12,
                    slat_sampling_steps=12,
                    slat_guidance_strength=3.0,
                )

            assert exc_info.value.code == "unauthorized"
            assert exc_info.value.message == "Invalid key"

    def test_generate_raises_file_not_found_for_missing_image(self) -> None:
        """generate() should raise FileNotFoundError for missing image."""
        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with pytest.raises(FileNotFoundError):
            client.generate(
                image_path="/nonexistent/image.png",
                seed=42,
                ss_sampling_steps=12,
                slat_sampling_steps=12,
                slat_guidance_strength=3.0,
            )


class TestTrellisAPIClientExtractGLB:
    """Tests for TrellisAPIClient.extract_glb()."""

    def test_extract_glb_sends_correct_request(self, tmp_path: Path) -> None:
        """extract_glb() should send POST with correct headers and payload."""
        output_path = tmp_path / "output.glb"

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"glb": base64.b64encode(b"glb data").decode()},
            )

            client.extract_glb(
                state="c3RhdGU=",  # Base64 compressed state from generate()
                mesh_simplify_ratio=0.95,
                texture_size=1024,
                output_path=str(output_path),
            )

            # Verify call
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            # Check method and URL
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "https://example.com/extract_glb"

            # Check headers
            assert call_args[1]["headers"]["X-API-Key"] == "sk_dev_test123"

            # Check payload
            payload = call_args[1]["json"]
            assert payload["state"] == "c3RhdGU="  # State passed through
            assert payload["mesh_simplify_ratio"] == 0.95
            assert payload["texture_size"] == 1024

    def test_extract_glb_writes_output_file(self, tmp_path: Path) -> None:
        """extract_glb() should write GLB data to output path."""
        output_path = tmp_path / "output.glb"

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        glb_data = b"fake glb binary data"
        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"glb": base64.b64encode(glb_data).decode()},
            )

            result = client.extract_glb(
                state="c3RhdGU=",
                mesh_simplify_ratio=0.95,
                texture_size=1024,
                output_path=str(output_path),
            )

            assert output_path.exists()
            assert output_path.read_bytes() == glb_data
            assert result == str(output_path)

    def test_extract_glb_raises_api_error_on_error_response(self, tmp_path: Path) -> None:
        """extract_glb() should raise APIError on error response."""
        output_path = tmp_path / "output.glb"

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"error": {"code": "cuda_oom", "message": "Out of memory"}},
            )

            with pytest.raises(APIError) as exc_info:
                client.extract_glb(
                    state="c3RhdGU=",
                    mesh_simplify_ratio=0.95,
                    texture_size=1024,
                    output_path=str(output_path),
                )

            assert exc_info.value.code == "cuda_oom"


class TestTrellisAPIClientExtractGaussian:
    """Tests for TrellisAPIClient.extract_gaussian()."""

    def test_extract_gaussian_sends_correct_request(self, tmp_path: Path) -> None:
        """extract_gaussian() should send POST with correct headers and payload."""
        output_path = tmp_path / "output.ply"

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"ply": base64.b64encode(b"ply data").decode()},
            )

            client.extract_gaussian(
                state="c3RhdGU=",  # Base64 compressed state from generate()
                output_path=str(output_path),
            )

            # Verify call
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            # Check method and URL
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "https://example.com/extract_gaussian"

            # Check payload
            payload = call_args[1]["json"]
            assert payload["state"] == "c3RhdGU="

    def test_extract_gaussian_writes_output_file(self, tmp_path: Path) -> None:
        """extract_gaussian() should write PLY data to output path."""
        output_path = tmp_path / "output.ply"

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        ply_data = b"fake ply binary data"
        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"ply": base64.b64encode(ply_data).decode()},
            )

            result = client.extract_gaussian(
                state="c3RhdGU=",
                output_path=str(output_path),
            )

            assert output_path.exists()
            assert output_path.read_bytes() == ply_data
            assert result == str(output_path)


class TestColdStartDetection:
    """Tests for cold start detection."""

    def test_last_request_elapsed_is_none_initially(self) -> None:
        """last_request_elapsed should be None before any request."""
        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )
        assert client.last_request_elapsed is None

    def test_last_request_elapsed_is_set_after_generate(self, tmp_path: Path) -> None:
        """last_request_elapsed should be set after generate() call."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            client.generate(
                image_path=str(image_path),
                seed=42,
                ss_sampling_steps=12,
                slat_sampling_steps=12,
                slat_guidance_strength=3.0,
            )

            assert client.last_request_elapsed is not None
            assert client.last_request_elapsed >= 0

    def test_was_cold_start_true_for_slow_request(self, tmp_path: Path) -> None:
        """was_cold_start() should return True for slow requests."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        # Simulate slow response by patching time
        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            # Force a slow elapsed time
            with patch("time.perf_counter", side_effect=[0.0, 10.0]):  # 10 seconds
                client.generate(
                    image_path=str(image_path),
                    seed=42,
                    ss_sampling_steps=12,
                    slat_sampling_steps=12,
                    slat_guidance_strength=3.0,
                )

            assert client.was_cold_start(threshold=5.0) is True

    def test_was_cold_start_false_for_fast_request(self, tmp_path: Path) -> None:
        """was_cold_start() should return False for fast requests."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
            )

            # Force a fast elapsed time
            with patch("time.perf_counter", side_effect=[0.0, 0.5]):  # 0.5 seconds
                client.generate(
                    image_path=str(image_path),
                    seed=42,
                    ss_sampling_steps=12,
                    slat_sampling_steps=12,
                    slat_guidance_strength=3.0,
                )

            assert client.was_cold_start(threshold=5.0) is False

    def test_was_cold_start_false_when_no_request_made(self) -> None:
        """was_cold_start() should return False when no request made."""
        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )
        assert client.was_cold_start(threshold=5.0) is False

    def test_last_request_elapsed_is_set_after_extract_glb(self, tmp_path: Path) -> None:
        """last_request_elapsed should be set after extract_glb() call."""
        output_path = tmp_path / "output.glb"

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"glb": base64.b64encode(b"glb data").decode()},
            )

            client.extract_glb(
                state="c3RhdGU=",
                mesh_simplify_ratio=0.95,
                texture_size=1024,
                output_path=str(output_path),
            )

            assert client.last_request_elapsed is not None
            assert client.last_request_elapsed >= 0

    def test_last_request_elapsed_is_set_after_extract_gaussian(self, tmp_path: Path) -> None:
        """last_request_elapsed should be set after extract_gaussian() call."""
        output_path = tmp_path / "output.ply"

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"ply": base64.b64encode(b"ply data").decode()},
            )

            client.extract_gaussian(
                state="c3RhdGU=",
                output_path=str(output_path),
            )

            assert client.last_request_elapsed is not None
            assert client.last_request_elapsed >= 0


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    def test_retries_on_connection_error(self, tmp_path: Path) -> None:
        """Should retry on ConnectionError."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            # First call fails with ConnectionError, second succeeds
            mock_request.side_effect = [
                requests.exceptions.ConnectionError("Connection failed"),
                MagicMock(
                    status_code=200,
                    json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
                ),
            ]

            with patch("time.sleep"):  # Skip actual delays
                result = client.generate(
                    image_path=str(image_path),
                    seed=42,
                    ss_sampling_steps=12,
                    slat_sampling_steps=12,
                    slat_guidance_strength=3.0,
                )

            assert "state" in result
            assert mock_request.call_count == 2

    def test_retries_on_timeout(self, tmp_path: Path) -> None:
        """Should retry on Timeout."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            # First call times out, second succeeds
            mock_request.side_effect = [
                requests.exceptions.Timeout("Request timed out"),
                MagicMock(
                    status_code=200,
                    json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
                ),
            ]

            with patch("time.sleep"):
                result = client.generate(
                    image_path=str(image_path),
                    seed=42,
                    ss_sampling_steps=12,
                    slat_sampling_steps=12,
                    slat_guidance_strength=3.0,
                )

            assert "state" in result
            assert mock_request.call_count == 2

    def test_raises_after_max_retries(self, tmp_path: Path) -> None:
        """Should raise after exhausting retries."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            # All calls fail
            mock_request.side_effect = requests.exceptions.ConnectionError("Connection failed")

            with patch("time.sleep"):
                with pytest.raises(requests.exceptions.ConnectionError):
                    client.generate(
                        image_path=str(image_path),
                        seed=42,
                        ss_sampling_steps=12,
                        slat_sampling_steps=12,
                        slat_guidance_strength=3.0,
                    )

            # Should have tried 3 times (1 initial + 2 retries)
            assert mock_request.call_count == 3

    def test_does_not_retry_api_errors(self, tmp_path: Path) -> None:
        """Should not retry on API errors like unauthorized."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_bad",
        )

        with patch("requests.request") as mock_request:
            # API error response
            mock_request.return_value = MagicMock(
                status_code=200,
                json=lambda: {"error": {"code": "unauthorized", "message": "Invalid key"}},
            )

            with pytest.raises(APIError):
                client.generate(
                    image_path=str(image_path),
                    seed=42,
                    ss_sampling_steps=12,
                    slat_sampling_steps=12,
                    slat_guidance_strength=3.0,
                )

            # Should only call once - no retry for API errors
            assert mock_request.call_count == 1

    def test_exponential_backoff_delays(self, tmp_path: Path) -> None:
        """Should use exponential backoff for delays."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake png data")

        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        sleep_calls = []

        with patch("requests.request") as mock_request:
            mock_request.side_effect = [
                requests.exceptions.ConnectionError("Fail 1"),
                requests.exceptions.ConnectionError("Fail 2"),
                MagicMock(
                    status_code=200,
                    json=lambda: {"state": "c3RhdGU=", "video": "dmlkZW8="},
                ),
            ]

            with patch("time.sleep", side_effect=lambda x: sleep_calls.append(x)):
                client.generate(
                    image_path=str(image_path),
                    seed=42,
                    ss_sampling_steps=12,
                    slat_sampling_steps=12,
                    slat_guidance_strength=3.0,
                )

            # First retry: 1s, second retry: 2s (exponential backoff)
            assert len(sleep_calls) == 2
            assert sleep_calls[0] == 1.0
            assert sleep_calls[1] == 2.0


class TestHealthCheck:
    """Tests for health_check() method."""

    def test_health_check_returns_true_on_success(self) -> None:
        """health_check() should return True when service responds."""
        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=200)

            assert client.health_check() is True

    def test_health_check_returns_true_on_error_response(self) -> None:
        """health_check() should return True even on HTTP error (service is reachable)."""
        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=500)

            assert client.health_check() is True

    def test_health_check_returns_false_on_connection_error(self) -> None:
        """health_check() should return False when service is unreachable."""
        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            mock_request.side_effect = requests.exceptions.ConnectionError("Connection failed")

            assert client.health_check() is False

    def test_health_check_returns_false_on_timeout(self) -> None:
        """health_check() should return False on timeout."""
        client = TrellisAPIClient(
            base_url="https://example.com",
            api_key="sk_dev_test123",
        )

        with patch("requests.request") as mock_request:
            mock_request.side_effect = requests.exceptions.Timeout("Request timed out")

            assert client.health_check() is False


class TestAPIError:
    """Tests for APIError exception."""

    def test_api_error_stores_code_and_message(self) -> None:
        """APIError should store code and message."""
        err = APIError("test_code", "test message")
        assert err.code == "test_code"
        assert err.message == "test message"

    def test_api_error_str_includes_code_and_message(self) -> None:
        """APIError str should include code and message."""
        err = APIError("test_code", "test message")
        assert str(err) == "test_code: test message"
