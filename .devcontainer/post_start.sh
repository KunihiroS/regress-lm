#!/bin/bash
# post_start.sh
# This script runs after the container is created and attached.
# It tags the container's base image with the current timestamp.

set -e

# デバッグ情報を出力
echo "[post-start] Starting post-start script..."
echo "[post-start] Current user: $(whoami)"
echo "[post-start] Environment variables:"
printenv | sort

# 必要な環境変数のチェック
if [ -z "$IMAGE_NAME_BASE" ]; then
    echo "[post-start] Error: IMAGE_NAME_BASE environment variable is not set. Check devcontainer.json." >&2
    exit 1
fi

# --- Script Body ---
echo "[post-start] Starting image tagging process..."

# Podman ソケットの存在を確認
PODMAN_SOCKET="/run/user/1000/podman/podman.sock"
if [ ! -S "$PODMAN_SOCKET" ]; then
    echo "[post-start] Warning: Podman socket not found at $PODMAN_SOCKET" >&2
    echo "[post-start] Trying alternative location..."
    PODMAN_SOCKET="/var/run/podman/podman.sock"
    if [ ! -S "$PODMAN_SOCKET" ]; then
        echo "[post-start] Error: Podman socket not found at $PODMAN_SOCKET" >&2
        echo "[post-start] Please ensure Podman is running and the socket is accessible." >&2
        exit 1
    fi
fi

export PODMAN_HOST="unix://$PODMAN_SOCKET"
echo "[post-start] Using Podman socket at $PODMAN_SOCKET"

# コンテナ情報を取得
echo "[post-start] Container information:"
CONTAINER_ID=$(hostname)
if [ -z "$CONTAINER_ID" ]; then
    echo "[post-start] Error: Could not determine container ID from hostname." >&2
    exit 1
fi

# Get the full image ID (SHA) from the container ID using the host's Podman daemon.
IMAGE_ID=$(sudo podman --url "$PODMAN_HOST" inspect --format '{{.Image}}' "$CONTAINER_ID")
if [ -z "$IMAGE_ID" ]; then
    echo "[post-start] Error: Could not determine image ID from container '${CONTAINER_ID}'." >&2
    exit 1
fi

# Generate a timestamp tag for today's date (e.g., YYYYMMDD).
TODAY_TAG_DATE=$(date +%Y%m%d)
NEW_IMAGE_NAME="${IMAGE_NAME_BASE}:${TODAY_TAG_DATE}"

# Check if an image with today's tag already exists to ensure idempotency.
EXISTING_TAG=$(sudo podman --url "$PODMAN_HOST" image list --format "{{.Repository}}:{{.Tag}}" | grep "^${NEW_IMAGE_NAME}$" || true)
if [ -n "$EXISTING_TAG" ]; then
    echo "[post-start] Image for today (${TODAY_TAG_DATE}) already has a timestamp tag. Skipping."
    exit 0
fi

# Apply the new timestamp tag to the image.
sudo podman --url "$PODMAN_HOST" tag "$IMAGE_ID" "$NEW_IMAGE_NAME"

echo "[post-start] Successfully tagged image as ${NEW_IMAGE_NAME}"

# --- End of Script ---
