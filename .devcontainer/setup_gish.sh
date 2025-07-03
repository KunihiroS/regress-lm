#!/bin/bash
set -e

# This script is run as root by devcontainer.json's onCreateCommand

# 1. Define paths
GISH_INSTALL_DIR="/usr/local/share/gish-tools"
GISH_SRC_DIR="/tmp/gishscript-src"
GISH_SCRIPT_PATH="${GISH_SRC_DIR}/gish.sh"
VENV_PATH="${GISH_INSTALL_DIR}/.venv"

# 2. Clone the repository
echo "Cloning gishscript repository..."
# The python and common-utils features should have installed git already.
git clone https://github.com/kunihiros/gishscript.git "${GISH_SRC_DIR}"

# 3. Patch the gish.sh script to make it container-aware
echo "Patching gish.sh for container environment..."
sed -i "s|\\\$HOME/.local/bin/gish-tools/venv|${VENV_PATH}|g" "$GISH_SCRIPT_PATH"
sed -i "/^TOOLS_DIR=/c\TOOLS_DIR=\"${GISH_INSTALL_DIR}\"" "$GISH_SCRIPT_PATH"
sed -i "s|/venv/bin/python3|/.venv/bin/python3|g" "$GISH_SCRIPT_PATH"
sed -i "/^DEBUG_FILE=/c\DEBUG_FILE=\"/tmp/gish_debug.log\"" "$GISH_SCRIPT_PATH"
echo "Patching complete."

# 4. Create installation directory and copy files
echo "Setting up gish installation directory..."
mkdir -p "${GISH_INSTALL_DIR}"
cp "${GISH_SCRIPT_PATH}" "${GISH_INSTALL_DIR}/gish.sh"
cp "${GISH_SRC_DIR}/generate_commit_message.py" "${GISH_INSTALL_DIR}/"
cp "${GISH_SRC_DIR}/requirements.txt" "${GISH_INSTALL_DIR}/"

# 5. Create Python virtual environment and install dependencies with uv
echo "Creating Python virtual environment with uv..."
uv venv -p python3 "${VENV_PATH}"
echo "Installing Python dependencies with uv..."
uv pip install --python "${VENV_PATH}/bin/python" --no-cache-dir -r "${GISH_INSTALL_DIR}/requirements.txt"

# 6. Create the .env file from the temporary file
echo "Creating .env file for gish..."
if [ ! -f "${GISH_INSTALL_DIR}/.env" ]; then
    echo "Final .env file not found. Proceeding with creation..."
    if [ -f "${PWD}/.devcontainer/.env.tmp" ]; then
        echo "Creating .env file from .env.tmp..."
        mv "${PWD}/.devcontainer/.env.tmp" "${GISH_INSTALL_DIR}/.env"
        chmod 600 "${GISH_INSTALL_DIR}/.env"
    else
        echo "Warning: .env.tmp not found. gish may not work without an API key."
    fi
else
    echo "Final .env file already exists. Skipping creation."
    # 既存の.envがある場合は、一時ファイルは不要なので削除
    if [ -f "${PWD}/.devcontainer/.env.tmp" ]; then
        rm "${PWD}/.devcontainer/.env.tmp"
    fi
fi

# 7. Install the main gish script to a directory in PATH
echo "Installing gish command..."
ln -sf "${GISH_INSTALL_DIR}/gish.sh" /usr/local/bin/gish
chmod +x "${GISH_INSTALL_DIR}/gish.sh"

# 8. Change ownership to the vscode user to prevent runtime permission issues
echo "Setting permissions for vscode user..."
chown -R vscode:vscode "${GISH_INSTALL_DIR}"

# 9. Clean up
echo "Cleaning up temporary files..."
rm -rf "${GISH_SRC_DIR}"

echo "gish setup complete."
