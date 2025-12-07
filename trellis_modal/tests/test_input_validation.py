"""
Tests for input validation hardening.

Tests image size limits, dimension limits, and format validation
to prevent abuse and protect GPU resources.
"""

import base64
from io import BytesIO

from PIL import Image

from trellis_modal.service.service import _parse_generate_request


def _make_image_b64(width: int, height: int, format: str = "PNG") -> str:
    """Create a base64-encoded test image."""
    img = Image.new("RGB", (width, height), color="red")
    buffer = BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestImageSizeLimits:
    """Tests for image payload size limits."""

    def test_rejects_oversized_base64_payload(self) -> None:
        """Should reject base64 payload larger than limit."""
        # Create ~15MB payload (larger than 10MB limit)
        large_b64 = "A" * (15 * 1024 * 1024)
        request = {"image": large_b64}
        result = _parse_generate_request(request)
        assert "error" in result
        assert result["error"]["code"] == "validation_error"
        assert "size" in result["error"]["message"].lower()

    def test_accepts_reasonable_size_payload(self) -> None:
        """Should accept payload under size limit."""
        request = {"image": _make_image_b64(512, 512)}
        result = _parse_generate_request(request)
        # Should return GenerateParams, not error dict
        assert not isinstance(result, dict) or "error" not in result


class TestImageDimensionLimits:
    """Tests for image dimension limits."""

    def test_rejects_oversized_dimensions(self) -> None:
        """Should reject images with dimensions exceeding limit."""
        # 5000x5000 is too large
        request = {"image": _make_image_b64(5000, 5000)}
        result = _parse_generate_request(request)
        assert "error" in result
        assert result["error"]["code"] == "validation_error"
        assert "dimension" in result["error"]["message"].lower()

    def test_accepts_reasonable_dimensions(self) -> None:
        """Should accept images with reasonable dimensions."""
        request = {"image": _make_image_b64(1024, 1024)}
        result = _parse_generate_request(request)
        assert not isinstance(result, dict) or "error" not in result


class TestImageFormatValidation:
    """Tests for image format validation."""

    def test_accepts_png_format(self) -> None:
        """Should accept PNG images."""
        request = {"image": _make_image_b64(256, 256, "PNG")}
        result = _parse_generate_request(request)
        assert not isinstance(result, dict) or "error" not in result

    def test_accepts_jpeg_format(self) -> None:
        """Should accept JPEG images."""
        request = {"image": _make_image_b64(256, 256, "JPEG")}
        result = _parse_generate_request(request)
        assert not isinstance(result, dict) or "error" not in result

    def test_rejects_non_image_data(self) -> None:
        """Should reject data that isn't a valid image."""
        # Random bytes that aren't an image
        fake_data = base64.b64encode(b"not an image file").decode("utf-8")
        request = {"image": fake_data}
        result = _parse_generate_request(request)
        assert "error" in result
        assert result["error"]["code"] == "validation_error"
