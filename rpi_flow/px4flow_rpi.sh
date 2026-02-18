#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${ROOT_DIR}/config_sitl.json"
if [ ! -f "${DEFAULT_CONFIG}" ]; then
  DEFAULT_CONFIG="${ROOT_DIR}/config.json"
fi
CONFIG_PATH="${1:-${DEFAULT_CONFIG}}"

echo "[px4flow_rpi] using config: ${CONFIG_PATH}"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo -n"
  fi
fi

detect_camera_backend() {
  python3 - "$1" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cam = data.get("camera", {}) if isinstance(data, dict) else {}
    backend = cam.get("backend", "") if isinstance(cam, dict) else ""
    print(str(backend).strip().lower())
except Exception:
    print("")
PY
}

maybe_source_ros2() {
  if [ "$1" != "ros2" ]; then
    return
  fi
  local ros_setup="${ROS_SETUP_BASH:-}"
  if [ -z "${ros_setup}" ]; then
    for c in \
      /opt/ros/jazzy/setup.bash \
      /opt/ros/humble/setup.bash \
      /opt/ros/iron/setup.bash \
      /opt/ros/rolling/setup.bash; do
      if [ -f "${c}" ]; then
        ros_setup="${c}"
        break
      fi
    done
  fi
  if [ -n "${ros_setup}" ] && [ -f "${ros_setup}" ]; then
    # shellcheck disable=SC1090
    source "${ros_setup}"
    echo "[px4flow_rpi] sourced ROS2 setup: ${ros_setup}"
  else
    echo "[px4flow_rpi] ROS2 backend selected, assuming ROS env is already sourced"
  fi
}

maybe_source_ros1() {
  if [ "$1" != "ros1" ]; then
    return
  fi
  local ros_setup="${ROS1_SETUP_BASH:-}"
  if [ -z "${ros_setup}" ]; then
    for c in \
      /opt/ros/noetic/setup.bash \
      /opt/ros/melodic/setup.bash \
      /opt/ros/kinetic/setup.bash; do
      if [ -f "${c}" ]; then
        ros_setup="${c}"
        break
      fi
    done
  fi
  if [ -n "${ros_setup}" ] && [ -f "${ros_setup}" ]; then
    # shellcheck disable=SC1090
    source "${ros_setup}"
    echo "[px4flow_rpi] sourced ROS1 setup: ${ros_setup}"
  else
    echo "[px4flow_rpi] ROS1 backend selected, assuming ROS env is already sourced"
  fi
}

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

CAM_BACKEND="$(detect_camera_backend "${CONFIG_PATH}")"
maybe_source_ros2 "${CAM_BACKEND}"
maybe_source_ros1 "${CAM_BACKEND}"

if [ "${CAM_BACKEND}" = "picamera2" ] || [ "${CAM_BACKEND}" = "opencv" ]; then
  kill_camera_users
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
exec python3 -m px4flow_rpi.main --config "${CONFIG_PATH}"
