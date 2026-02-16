from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationResult:
    camera_matrix: Any  # np.ndarray (3x3)
    dist_coeffs: Any  # np.ndarray (N,)
    rms: float
    source: str


def _as_camera_matrix(raw: Any) -> Any:
    import numpy as np  # type: ignore

    k = np.asarray(raw, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {k.shape}")
    return k


def _as_dist_coeffs(raw: Any) -> Any:
    import numpy as np  # type: ignore

    d = np.asarray(raw, dtype=np.float64).reshape(-1)
    if d.size < 5:
        raise ValueError("dist_coeffs must have at least 5 elements (k1,k2,p1,p2,k3)")
    return d


def _resolve_path(raw_path: str, base_dir: str) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p
    if base_dir:
        return Path(base_dir).expanduser() / p
    return p


def _load_cached_calibration(path: Path) -> CalibrationResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("calibration cache must be an object")
    k = _as_camera_matrix(data.get("camera_matrix", data.get("K")))
    d = _as_dist_coeffs(data.get("dist_coeffs", data.get("dist")))
    rms = float(data.get("rms", 0.0))
    return CalibrationResult(camera_matrix=k, dist_coeffs=d, rms=rms, source=f"cache:{path}")


def _save_cached_calibration(path: Path, calib: CalibrationResult) -> None:
    data = {
        "camera_matrix": calib.camera_matrix.tolist(),
        "dist_coeffs": calib.dist_coeffs.reshape(-1).tolist(),
        "rms": float(calib.rms),
        "source": calib.source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _calibrate_zhang(cfg: dict[str, Any], width: int, height: int) -> CalibrationResult:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    images_glob = str(cfg.get("images_glob", "")).strip()
    if not images_glob:
        raise ValueError("camera.calibration.images_glob is required for zhang calibration")

    base_dir = str(cfg.get("base_dir", "")).strip()
    pattern = _resolve_path(images_glob, base_dir)
    image_paths = sorted(glob.glob(str(pattern)))
    if not image_paths:
        raise ValueError(f"no calibration images matched: {pattern}")

    chessboard_size = cfg.get("chessboard_size", [9, 6])
    if not isinstance(chessboard_size, (list, tuple)) or len(chessboard_size) != 2:
        raise ValueError("camera.calibration.chessboard_size must be [cols, rows]")
    cols = int(chessboard_size[0])
    rows = int(chessboard_size[1])
    if cols < 3 or rows < 3:
        raise ValueError("camera.calibration.chessboard_size is too small")

    square_size = float(cfg.get("square_size", 1.0))
    min_images = max(3, int(cfg.get("min_images", 8)))
    subpix_window = max(2, int(cfg.get("subpix_window", 11)))
    use_sb = bool(cfg.get("use_find_chessboard_sb", True))

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []
    used_files = 0
    target_size = (int(width), int(height))
    term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape[1] != target_size[0] or img.shape[0] != target_size[1]:
            continue
        if use_sb:
            ok, corners = cv2.findChessboardCornersSB(img, (cols, rows), None)
        else:
            ok, corners = cv2.findChessboardCorners(img, (cols, rows), None)
        if not ok or corners is None:
            continue
        corners2 = cv2.cornerSubPix(
            img,
            corners,
            (subpix_window, subpix_window),
            (-1, -1),
            term,
        )
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        used_files += 1

    if used_files < min_images:
        raise ValueError(
            f"insufficient calibration images with valid chessboard detections: {used_files} < {min_images}"
        )

    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        target_size,
        None,
        None,
    )
    if not np.isfinite(rms):
        raise ValueError("calibrateCamera returned non-finite RMS")

    # Keep at least the classic 5-parameter distortion model.
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if dist_coeffs.size < 5:
        padded = np.zeros((5,), dtype=np.float64)
        padded[: dist_coeffs.size] = dist_coeffs
        dist_coeffs = padded

    return CalibrationResult(
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        dist_coeffs=dist_coeffs,
        rms=float(rms),
        source=f"zhang:{pattern}",
    )


def load_camera_calibration(camera_cfg: dict[str, Any], width: int, height: int) -> CalibrationResult | None:
    calib_cfg = camera_cfg.get("calibration", {}) if isinstance(camera_cfg.get("calibration", {}), dict) else {}
    if not bool(calib_cfg.get("enabled", False)):
        return None

    mode = str(calib_cfg.get("mode", "precomputed")).strip().lower()
    base_dir = str(calib_cfg.get("base_dir", "")).strip()
    cache_file = str(calib_cfg.get("cache_file", "")).strip()

    if cache_file and bool(calib_cfg.get("prefer_cache", True)):
        cache_path = _resolve_path(cache_file, base_dir)
        if cache_path.exists():
            return _load_cached_calibration(cache_path)

    if mode in ("precomputed", "load"):
        k_raw = calib_cfg.get("camera_matrix", calib_cfg.get("K"))
        d_raw = calib_cfg.get("dist_coeffs", calib_cfg.get("dist"))
        if k_raw is None or d_raw is None:
            raise ValueError("precomputed calibration requires camera_matrix/K and dist_coeffs/dist")
        return CalibrationResult(
            camera_matrix=_as_camera_matrix(k_raw),
            dist_coeffs=_as_dist_coeffs(d_raw),
            rms=float(calib_cfg.get("rms", 0.0)),
            source="precomputed",
        )

    if mode in ("zhang", "calibrate"):
        calib = _calibrate_zhang(calib_cfg, width=width, height=height)
        if cache_file:
            cache_path = _resolve_path(cache_file, base_dir)
            _save_cached_calibration(cache_path, calib)
        return calib

    raise ValueError(f"unsupported calibration mode: {mode}")
