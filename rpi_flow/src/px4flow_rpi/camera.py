from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .camera_calibration import CalibrationResult, load_camera_calibration


@dataclass
class Frame:
    gray: Any  # numpy array
    t_monotonic: float


class Camera:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._backend = cfg.get("backend", "picamera2")
        self._width = int(cfg.get("width", 640))
        self._height = int(cfg.get("height", 480))
        self._fps = float(cfg.get("fps", 60))
        self._controls = cfg.get("controls", {}) if isinstance(cfg.get("controls", {}), dict) else {}
        self._roi = cfg.get("roi", None)

        self._impl = None
        self._calibration: CalibrationResult | None = None
        self._camera_matrix = None
        self._dist_coeffs = None
        self._undistort_map1 = None
        self._undistort_map2 = None
        self._undistort_size: tuple[int, int] | None = None
        self._rectify_alpha = 0.0
        self._start()
        self._load_calibration()

    def _start(self) -> None:
        if self._backend == "picamera2":
            try:
                from picamera2 import Picamera2  # type: ignore
            except Exception:
                self._backend = "opencv"
            else:
                picam2 = Picamera2()
                controls = dict(self._controls)
                if "FrameDurationLimits" not in controls:
                    controls["FrameRate"] = self._fps
                video_config = picam2.create_video_configuration(
                    main={"size": (self._width, self._height), "format": "YUV420"},
                    controls=controls,
                )
                picam2.configure(video_config)
                picam2.start()
                self._impl = ("picamera2", picam2)
                return

        if self._backend == "opencv":
            import cv2  # type: ignore

            dev = int(self._cfg.get("device", 0))
            cap = cv2.VideoCapture(dev)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            self._impl = ("opencv", cap)
            return

        raise ValueError(f"unknown camera backend: {self._backend}")

    def _load_calibration(self) -> None:
        calib_cfg = self._cfg.get("calibration", {}) if isinstance(self._cfg.get("calibration", {}), dict) else {}
        self._rectify_alpha = float(calib_cfg.get("rectify_alpha", 0.0))
        self._calibration = load_camera_calibration(self._cfg, width=self._width, height=self._height)
        if self._calibration is None:
            return
        self._dist_coeffs = self._calibration.dist_coeffs
        self._rebuild_undistort_maps(width=self._width, height=self._height)
        print(
            f"camera calibration loaded source={self._calibration.source} "
            f"rms={self._calibration.rms:.4f}"
        )

    def _rebuild_undistort_maps(self, width: int, height: int) -> None:
        if self._calibration is None:
            return
        import cv2  # type: ignore

        image_size = (int(width), int(height))
        k = self._calibration.camera_matrix
        d = self._calibration.dist_coeffs
        new_k, _valid_roi = cv2.getOptimalNewCameraMatrix(k, d, image_size, self._rectify_alpha, image_size)
        map1, map2 = cv2.initUndistortRectifyMap(k, d, None, new_k, image_size, cv2.CV_16SC2)
        self._camera_matrix = new_k
        self._undistort_map1 = map1
        self._undistort_map2 = map2
        self._undistort_size = image_size

    def _apply_undistort(self, gray: Any) -> Any:
        if self._calibration is None:
            return gray
        import cv2  # type: ignore

        h, w = gray.shape[:2]
        if self._undistort_size != (w, h):
            self._rebuild_undistort_maps(width=w, height=h)
        if self._undistort_map1 is None or self._undistort_map2 is None:
            return gray
        return cv2.remap(gray, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)

    @staticmethod
    def _crop_roi(gray: Any, roi: Any) -> Any:
        if isinstance(roi, list) and len(roi) == 4:
            x, y, w, h = [int(v) for v in roi]
            if w > 0 and h > 0:
                return gray[y : y + h, x : x + w]
        return gray

    def camera_matrix(self) -> Any | None:
        import numpy as np  # type: ignore

        if self._camera_matrix is None:
            return None
        k = np.asarray(self._camera_matrix, dtype=np.float64).copy()
        if isinstance(self._roi, list) and len(self._roi) == 4:
            x, y, _w, _h = [int(v) for v in self._roi]
            k[0, 2] -= float(x)
            k[1, 2] -= float(y)
        return k

    def read(self) -> Frame:
        backend, impl = self._impl

        if backend == "picamera2":
            import numpy as np  # type: ignore

            arr = impl.capture_array("main")  # YUV420 or mono
            if getattr(arr, "ndim", 0) == 3:
                gray = arr[:, :, 0]
            elif getattr(arr, "ndim", 0) == 2:
                # Mono sensors or planar YUV420 can come back 2D; use luma plane.
                if arr.shape[0] >= self._height:
                    gray = arr[: self._height, :]
                else:
                    gray = arr
            else:
                raise RuntimeError(f"unexpected frame shape: {getattr(arr, 'shape', None)}")
            gray = self._apply_undistort(gray)
            gray = self._crop_roi(gray, self._roi)
            gray = np.ascontiguousarray(gray)
            t = time.monotonic()
            return Frame(gray=gray, t_monotonic=t)

        if backend == "opencv":
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            ok, frame = impl.read()
            if not ok:
                raise RuntimeError("camera read failed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = self._apply_undistort(gray)
            gray = self._crop_roi(gray, self._roi)
            gray = np.ascontiguousarray(gray)
            t = time.monotonic()
            return Frame(gray=gray, t_monotonic=t)

        raise RuntimeError("camera not initialized")

    def close(self) -> None:
        if self._impl is None:
            return
        backend, impl = self._impl
        if backend == "opencv":
            impl.release()
        elif backend == "picamera2":
            impl.stop()
        self._impl = None
