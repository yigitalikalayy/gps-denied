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

detect_log_dir() {
  python3 - "$1" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging_cfg = data.get("logging", {}) if isinstance(data, dict) else {}
    flight_log = logging_cfg.get("flight_log", {}) if isinstance(logging_cfg, dict) else {}
    log_dir = flight_log.get("dir", "") if isinstance(flight_log, dict) else ""
    print(str(log_dir).strip())
except Exception:
    print("")
PY
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

kill_camera_users

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
LOG_DIR_REL="$(detect_log_dir "${CONFIG_PATH}")"
if [ -z "${LOG_DIR_REL}" ]; then
  LOG_DIR="${REPO_ROOT}/logs"
else
  if [[ "${LOG_DIR_REL}" = /* ]]; then
    LOG_DIR="${LOG_DIR_REL}"
  else
    LOG_DIR="${REPO_ROOT}/${LOG_DIR_REL}"
  fi
fi

mkdir -p "${LOG_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
max_log=0
shopt -s nullglob
for f in "${LOG_DIR}"/log_*.txt; do
  base="$(basename "${f}")"
  n="${base#log_}"
  n="${n%%_*}"
  if [[ "${n}" =~ ^[0-9]+$ ]]; then
    if ((10#${n} > max_log)); then
      max_log=$((10#${n}))
    fi
  fi
done
shopt -u nullglob
log_num=$((max_log + 1))
log_file="$(printf "%s/log_%04d_%s.txt" "${LOG_DIR}" "${log_num}" "${timestamp}")"
echo "[px4flow_rpi] logging to: ${log_file}"

prune_logs() {
  local keep="${1:-100}"
  local files=()
  local path
  shopt -s nullglob
  for path in "${LOG_DIR}"/log_*.txt; do
    files+=("${path}")
  done
  shopt -u nullglob
  if [ "${#files[@]}" -le "${keep}" ]; then
    return
  fi
  IFS=$'\n' read -r -d '' -a sorted < <(printf '%s\n' "${files[@]}" | sort && printf '\0')
  local remove_count=$(( ${#sorted[@]} - keep ))
  if [ "${remove_count}" -le 0 ]; then
    return
  fi
  for ((i=0; i<remove_count; i++)); do
    rm -f "${sorted[$i]}" || true
  done
}

prune_logs 100

python3 -m algorithms.optical_flow.main --config "${CONFIG_PATH}" 2>&1 | tee -a "${log_file}"
exit ${PIPESTATUS[0]}
