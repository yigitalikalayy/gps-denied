from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


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
        self._start()

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

    def read(self) -> Frame:
        backend, impl = self._impl
        t = time.monotonic()

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
            if isinstance(self._roi, list) and len(self._roi) == 4:
                x, y, w, h = [int(v) for v in self._roi]
                if w > 0 and h > 0:
                    gray = gray[y : y + h, x : x + w]
            gray = np.ascontiguousarray(gray)
            return Frame(gray=gray, t_monotonic=t)

        if backend == "opencv":
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            ok, frame = impl.read()
            if not ok:
                raise RuntimeError("camera read failed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if isinstance(self._roi, list) and len(self._roi) == 4:
                x, y, w, h = [int(v) for v in self._roi]
                if w > 0 and h > 0:
                    gray = gray[y : y + h, x : x + w]
            gray = np.ascontiguousarray(gray)
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
