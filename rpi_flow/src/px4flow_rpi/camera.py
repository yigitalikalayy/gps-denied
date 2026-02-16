from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from typing import Any

from .camera_calibration import CalibrationResult, load_camera_calibration


@dataclass
class Frame:
    gray: Any  # numpy array
    t_monotonic: float


class _Ros2ImageSource:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._topic = str(cfg.get("topic", "/camera/image_raw"))
        self._node_name = str(cfg.get("ros_node_name", "px4flow_rpi_camera"))
        self._timeout_s = max(0.05, float(cfg.get("ros_timeout_s", 1.0)))
        self._spin_timeout_s = max(0.001, float(cfg.get("ros_spin_timeout_s", 0.05)))
        self._queue: collections.deque[Frame] = collections.deque(maxlen=max(1, int(cfg.get("queue_size", 2))))
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        try:
            import rclpy  # type: ignore
            from rclpy.executors import SingleThreadedExecutor  # type: ignore
            from rclpy.node import Node  # type: ignore
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # type: ignore
            from sensor_msgs.msg import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "ros2 camera backend requires rclpy and sensor_msgs (source ROS 2 environment first)"
            ) from exc

        reliability_name = str(cfg.get("ros_qos_reliability", "best_effort")).strip().lower()
        reliability = ReliabilityPolicy.BEST_EFFORT
        if reliability_name == "reliable":
            reliability = ReliabilityPolicy.RELIABLE
        qos_depth = max(1, int(cfg.get("ros_qos_depth", 5)))
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=qos_depth, reliability=reliability)

        self._rclpy = rclpy
        self._owns_rclpy = not bool(rclpy.ok())
        if self._owns_rclpy:
            rclpy.init(args=None)

        self._node: Node = Node(self._node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        def _on_image(msg: Image) -> None:
            import numpy as np  # type: ignore

            try:
                gray = self._image_to_gray(msg)
                fr = Frame(gray=np.ascontiguousarray(gray), t_monotonic=time.monotonic())
            except Exception:
                return
            with self._cv:
                self._queue.append(fr)
                self._cv.notify_all()

        self._sub = self._node.create_subscription(Image, self._topic, _on_image, qos)
        self._thread = threading.Thread(target=self._spin_loop, name="ros2-camera", daemon=True)
        self._thread.start()

    @staticmethod
    def _reshape_rows(buf: Any, height: int, step: int):
        import numpy as np  # type: ignore

        arr = np.frombuffer(buf, dtype=np.uint8)
        expected = int(height) * int(step)
        if arr.size < expected:
            raise RuntimeError("ros2 image payload too small")
        return arr[:expected].reshape(int(height), int(step))

    @staticmethod
    def _image_to_gray(msg: Any):
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        if h <= 0 or w <= 0 or step <= 0:
            raise RuntimeError("invalid ros2 image dimensions")
        enc = str(msg.encoding).strip().lower()

        if enc in ("mono8", "8uc1"):
            rows = _Ros2ImageSource._reshape_rows(msg.data, h, step)
            if step < w:
                raise RuntimeError("invalid mono8 step")
            return rows[:, :w]

        if enc in ("bgr8", "rgb8"):
            if step < (w * 3):
                raise RuntimeError("invalid rgb/bgr step")
            rows = _Ros2ImageSource._reshape_rows(msg.data, h, step)
            img3 = rows[:, : (w * 3)].reshape(h, w, 3)
            code = cv2.COLOR_BGR2GRAY if enc == "bgr8" else cv2.COLOR_RGB2GRAY
            return cv2.cvtColor(img3, code)

        if enc in ("bgra8", "rgba8"):
            if step < (w * 4):
                raise RuntimeError("invalid rgba/bgra step")
            rows = _Ros2ImageSource._reshape_rows(msg.data, h, step)
            img4 = rows[:, : (w * 4)].reshape(h, w, 4)
            code = cv2.COLOR_BGRA2GRAY if enc == "bgra8" else cv2.COLOR_RGBA2GRAY
            return cv2.cvtColor(img4, code)

        if enc in ("mono16", "16uc1"):
            if step < (w * 2):
                raise RuntimeError("invalid mono16 step")
            dtype = np.dtype(">u2" if int(msg.is_bigendian) else "<u2")
            arr = np.frombuffer(msg.data, dtype=dtype)
            cols = step // 2
            if arr.size < (h * cols):
                raise RuntimeError("ros2 mono16 payload too small")
            img16 = arr[: h * cols].reshape(h, cols)[:, :w]
            return (img16 >> 8).astype(np.uint8)

        raise RuntimeError(f"unsupported ros2 image encoding: {enc}")

    def _spin_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=self._spin_timeout_s)
            except Exception:
                time.sleep(0.01)

    def read(self, timeout_s: float | None = None) -> Frame:
        timeout = self._timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout
        with self._cv:
            while (not self._queue) and (not self._stop.is_set()):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._cv.wait(timeout=remaining)
            if not self._queue:
                raise RuntimeError(f"ros2 image topic timeout: {self._topic}")
            fr = self._queue.pop()
            self._queue.clear()
            return fr

    def close(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            self._executor.shutdown(timeout_sec=0.2)
        except Exception:
            pass
        if self._owns_rclpy:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass


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

        if self._backend == "ros2":
            source = _Ros2ImageSource(self._cfg)
            self._impl = ("ros2", source)
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

        if backend == "ros2":
            import numpy as np  # type: ignore

            fr = impl.read(timeout_s=float(self._cfg.get("ros_timeout_s", 1.0)))
            gray = self._apply_undistort(fr.gray)
            gray = self._crop_roi(gray, self._roi)
            gray = np.ascontiguousarray(gray)
            return Frame(gray=gray, t_monotonic=float(fr.t_monotonic))

        raise RuntimeError("camera not initialized")

    def close(self) -> None:
        if self._impl is None:
            return
        backend, impl = self._impl
        if backend == "opencv":
            impl.release()
        elif backend == "picamera2":
            impl.stop()
        elif backend == "ros2":
            impl.close()
        self._impl = None
