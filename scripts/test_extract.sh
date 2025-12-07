#!/bin/bash
# Test script for TRELLIS /extract_glb and /extract_gaussian endpoints
#
# Usage:
#   ./scripts/test_extract.sh [ENDPOINT_BASE_URL] [STATE_FILE]
#
# Example:
#   ./scripts/test_extract.sh https://your-modal-url.modal.run ./test_output/state.bin
#
# The script:
# 1. Encodes the state file to base64
# 2. Sends a POST request to the extract_glb endpoint
# 3. Saves the GLB file to test_output/model.glb
# 4. Optionally tests extract_gaussian endpoint

set -e

# Default values
ENDPOINT_BASE_URL="${1:-http://localhost:8000}"
STATE_FILE="${2:-./test_output/state.bin}"
OUTPUT_DIR="./test_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}TRELLIS Extract Endpoint Test${NC}"
echo "================================"
echo "Endpoint Base: $ENDPOINT_BASE_URL"
echo "State File: $STATE_FILE"
echo ""

# Check if state file exists
if [ ! -f "$STATE_FILE" ]; then
    echo -e "${RED}Error: State file not found: $STATE_FILE${NC}"
    echo "Hint: Run test_generate.sh first to create the state file"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Encode state to base64
echo "Encoding state to base64..."
STATE_B64=$(base64 -w 0 "$STATE_FILE")

# ============================================
# Test /extract_glb
# ============================================
echo ""
echo -e "${YELLOW}Testing /extract_glb...${NC}"

# Create request body for GLB extraction
GLB_REQUEST_BODY=$(cat <<EOF
{
  "state": "$STATE_B64",
  "mesh_simplify_ratio": 0.95,
  "texture_size": 1024
}
EOF
)

echo "Sending request to $ENDPOINT_BASE_URL/extract_glb..."

# Send request and capture response
GLB_RESPONSE=$(curl -s -X POST "$ENDPOINT_BASE_URL/extract_glb" \
  -H "Content-Type: application/json" \
  -d "$GLB_REQUEST_BODY")

# Check for error
if echo "$GLB_RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
    ERROR_CODE=$(echo "$GLB_RESPONSE" | jq -r '.error.code')
    ERROR_MSG=$(echo "$GLB_RESPONSE" | jq -r '.error.message')
    echo -e "${RED}Error: $ERROR_CODE${NC}"
    echo "Message: $ERROR_MSG"
    exit 1
fi

# Check for glb field
if ! echo "$GLB_RESPONSE" | jq -e '.glb' > /dev/null 2>&1; then
    echo -e "${RED}Error: No 'glb' field in response${NC}"
    echo "Response: $GLB_RESPONSE"
    exit 1
fi

# Extract and decode GLB
echo "Extracting GLB..."
GLB_B64=$(echo "$GLB_RESPONSE" | jq -r '.glb')
echo "$GLB_B64" | base64 -d > "$OUTPUT_DIR/model.glb"
GLB_SIZE=$(stat -f%z "$OUTPUT_DIR/model.glb" 2>/dev/null || stat -c%s "$OUTPUT_DIR/model.glb")
echo -e "  ${GREEN}Saved: $OUTPUT_DIR/model.glb ($GLB_SIZE bytes)${NC}"

# ============================================
# Test /extract_gaussian
# ============================================
echo ""
echo -e "${YELLOW}Testing /extract_gaussian...${NC}"

# Create request body for Gaussian extraction
GAUSSIAN_REQUEST_BODY=$(cat <<EOF
{
  "state": "$STATE_B64"
}
EOF
)

echo "Sending request to $ENDPOINT_BASE_URL/extract_gaussian..."

# Send request and capture response
GAUSSIAN_RESPONSE=$(curl -s -X POST "$ENDPOINT_BASE_URL/extract_gaussian" \
  -H "Content-Type: application/json" \
  -d "$GAUSSIAN_REQUEST_BODY")

# Check for error
if echo "$GAUSSIAN_RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
    ERROR_CODE=$(echo "$GAUSSIAN_RESPONSE" | jq -r '.error.code')
    ERROR_MSG=$(echo "$GAUSSIAN_RESPONSE" | jq -r '.error.message')
    echo -e "${RED}Error: $ERROR_CODE${NC}"
    echo "Message: $ERROR_MSG"
    exit 1
fi

# Check for ply field
if ! echo "$GAUSSIAN_RESPONSE" | jq -e '.ply' > /dev/null 2>&1; then
    echo -e "${RED}Error: No 'ply' field in response${NC}"
    echo "Response: $GAUSSIAN_RESPONSE"
    exit 1
fi

# Extract and decode PLY
echo "Extracting PLY..."
PLY_B64=$(echo "$GAUSSIAN_RESPONSE" | jq -r '.ply')
echo "$PLY_B64" | base64 -d > "$OUTPUT_DIR/gaussian.ply"
PLY_SIZE=$(stat -f%z "$OUTPUT_DIR/gaussian.ply" 2>/dev/null || stat -c%s "$OUTPUT_DIR/gaussian.ply")
echo -e "  ${GREEN}Saved: $OUTPUT_DIR/gaussian.ply ($PLY_SIZE bytes)${NC}"

# ============================================
# Summary
# ============================================
echo ""
echo -e "${GREEN}Success!${NC}"
echo "Output files saved to $OUTPUT_DIR/"
echo ""
echo "Files created:"
echo "  - model.glb     (GLB mesh with texture)"
echo "  - gaussian.ply  (Gaussian splat)"
echo ""
echo "To view the GLB:"
echo "  - Open in Blender: blender --python-expr 'bpy.ops.import_scene.gltf(filepath=\"$OUTPUT_DIR/model.glb\")'"
echo "  - Or use online viewer: https://gltf-viewer.donmccurdy.com/"
