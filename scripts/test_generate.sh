#!/bin/bash
# Test script for TRELLIS /generate endpoint
#
# Usage:
#   ./scripts/test_generate.sh [ENDPOINT_URL] [IMAGE_PATH]
#
# Example:
#   ./scripts/test_generate.sh https://your-modal-url.modal.run/generate ./assets/example_image/typical_building_building.png
#
# The script:
# 1. Encodes the image to base64
# 2. Sends a POST request to the generate endpoint
# 3. Saves the response to output files (state.bin, video.mp4)

set -e

# Default values
ENDPOINT_URL="${1:-http://localhost:8000/generate}"
IMAGE_PATH="${2:-./assets/example_image/typical_building_building.png}"
OUTPUT_DIR="./test_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}TRELLIS Generate Endpoint Test${NC}"
echo "================================"
echo "Endpoint: $ENDPOINT_URL"
echo "Image: $IMAGE_PATH"
echo ""

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}Error: Image file not found: $IMAGE_PATH${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Encode image to base64
echo "Encoding image to base64..."
IMAGE_B64=$(base64 -w 0 "$IMAGE_PATH")

# Create request body
REQUEST_BODY=$(cat <<EOF
{
  "image": "$IMAGE_B64",
  "seed": 42,
  "ss_sampling_steps": 12,
  "slat_sampling_steps": 12,
  "slat_guidance_strength": 3.0
}
EOF
)

echo "Sending request to $ENDPOINT_URL..."
echo ""

# Send request and capture response
RESPONSE=$(curl -s -X POST "$ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY")

# Check for error
if echo "$RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
    ERROR_CODE=$(echo "$RESPONSE" | jq -r '.error.code')
    ERROR_MSG=$(echo "$RESPONSE" | jq -r '.error.message')
    echo -e "${RED}Error: $ERROR_CODE${NC}"
    echo "Message: $ERROR_MSG"
    exit 1
fi

# Check for state and video
if ! echo "$RESPONSE" | jq -e '.state' > /dev/null 2>&1; then
    echo -e "${RED}Error: No 'state' field in response${NC}"
    echo "Response: $RESPONSE"
    exit 1
fi

if ! echo "$RESPONSE" | jq -e '.video' > /dev/null 2>&1; then
    echo -e "${RED}Error: No 'video' field in response${NC}"
    exit 1
fi

# Extract and decode state
echo "Extracting state..."
STATE_B64=$(echo "$RESPONSE" | jq -r '.state')
echo "$STATE_B64" | base64 -d > "$OUTPUT_DIR/state.bin"
STATE_SIZE=$(stat -f%z "$OUTPUT_DIR/state.bin" 2>/dev/null || stat -c%s "$OUTPUT_DIR/state.bin")
echo "  Saved: $OUTPUT_DIR/state.bin ($STATE_SIZE bytes)"

# Extract and decode video
echo "Extracting video..."
VIDEO_B64=$(echo "$RESPONSE" | jq -r '.video')
echo "$VIDEO_B64" | base64 -d > "$OUTPUT_DIR/video.mp4"
VIDEO_SIZE=$(stat -f%z "$OUTPUT_DIR/video.mp4" 2>/dev/null || stat -c%s "$OUTPUT_DIR/video.mp4")
echo "  Saved: $OUTPUT_DIR/video.mp4 ($VIDEO_SIZE bytes)"

echo ""
echo -e "${GREEN}Success!${NC}"
echo "Output files saved to $OUTPUT_DIR/"
echo ""
echo "To view the video:"
echo "  open $OUTPUT_DIR/video.mp4  # macOS"
echo "  xdg-open $OUTPUT_DIR/video.mp4  # Linux"
