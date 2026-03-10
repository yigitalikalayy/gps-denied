#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/config.json}"

echo "[px4flow_rpi] using config: ${CONFIG_PATH}"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo -n"
  fi
fi

kill_camera_users() {
  echo "[px4flow_rpi] freeing camera resources..."

  if command -v fuser >/dev/null 2>&1; then
    ${SUDO} fuser -k /dev/video0 2>/dev/null || true
    ${SUDO} fuser -k /dev/media0 2>/dev/null || true
    ${SUDO} fuser -k /dev/vchiq 2>/dev/null || true
  fi

  if command -v pkill >/dev/null 2>&1; then
    ${SUDO} pkill -f "libcamera" 2>/dev/null || true
    ${SUDO} pkill -f "rpicam-" 2>/dev/null || true
    ${SUDO} pkill -f "picamera2" 2>/dev/null || true
    ${SUDO} pkill -f "gst-launch" 2>/dev/null || true
  fi

  sleep 0.5
}

kill_camera_users

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
exec python3 -m optical_flow.main --config "${CONFIG_PATH}"
