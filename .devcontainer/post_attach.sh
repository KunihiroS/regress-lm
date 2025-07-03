#!/bin/bash
# This script runs every time the container is attached to.

set -e

# デバッグ情報を出力
echo "[post-attach] Starting post-attach script..."
echo "[post-attach] Current directory: $(pwd)"
echo "[post-attach] Current user: $(whoami)"
echo "[post-attach] Environment variables:"
printenv | sort

# --- 1. .env.tmp Cleanup (Highest priority: contains secrets) ---
FINAL_ENV_FILE="/usr/local/share/gish-tools/.env"
TMP_ENV_FILE="${PWD}/.devcontainer/.env.tmp"

echo "[post-attach] Checking for temporary .env file..."
if [ -f "$FINAL_ENV_FILE" ] && [ -f "$TMP_ENV_FILE" ]; then
    echo "[post-attach] Setup is complete. Removing temporary .env file: $TMP_ENV_FILE"
    rm "$TMP_ENV_FILE"
else
    echo "[post-attach] Temporary .env file not found or setup not yet complete. No cleanup needed."
fi

# --- 2. Check GitHub Auth and run Git Fetch ---
echo "[post-attach] Checking GitHub authentication status..."
if gh auth status > /dev/null 2>&1; then
    echo "[post-attach] Already logged in to GitHub."
    echo "[post-attach] Forcing Git to use 'gh' as credential helper to override VS Code's default."
    # This is the crucial fix:
    # VS Code's Dev Container feature injects its own credential helper, which conflicts with gh.
    # We explicitly unset it and set the one from GitHub CLI to ensure git commands work.
    git config --global --unset-all credential.helper || true
    git config --global credential.helper '!/usr/bin/gh auth git-credential'

    echo "[post-attach] Fetching latest changes from all remotes..."
    # Now that the credential helper is correctly set, this should succeed.
    # We redirect stderr to /dev/null to suppress the harmless "fatal: could not read Username" message.
    if GIT_TERMINAL_PROMPT=0 git fetch --all --prune 2>/dev/null; then
        echo "[post-attach] Git fetch successful."
    else
        # This might happen if tokens expire, etc.
        echo "[post-attach] Git fetch failed even though logged in. Token might be invalid. Please try logging in again manually if needed."
    fi
else
    echo "[post-attach] Not logged into GitHub."
    # --- 2a. Attempt Git Fetch (will fail, but informs the user) ---
    echo "[post-attach] Attempting git fetch (expected to fail)..."
    if GIT_TERMINAL_PROMPT=0 git fetch --all --prune 2>/dev/null; then
        echo "[post-attach] Git fetch successful unexpectedly."
    else
        echo "[post-attach] Git fetch failed as expected. Please log in via the prompt below."
    fi

    # --- 2b. GitHub Auth Login (Interactive) ---
    echo "[post-attach] Starting GitHub login process..."
    # Use device flow for browser-less environments
    BROWSER= gh auth login
fi

echo "[post-attach] Script finished."
