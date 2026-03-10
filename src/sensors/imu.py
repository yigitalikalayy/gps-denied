from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GyroState:
    x_rad_s: float = 0.0
    y_rad_s: float = 0.0
    z_rad_s: float = 0.0
    temperature_c: float = 0.0
    t_wall: float = 0.0


@dataclass
class ImuYawSample:
    t_imu_s: float
    yaw_rad: float
    t_wall: float


class _BaseImuSubscriber:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._frame = str(cfg.get("frame", "ros_flu")).strip().lower()
        filter_cfg = cfg.get("filter", {}) if isinstance(cfg.get("filter", {}), dict) else {}
        self._gyro_max_rad_s = float(filter_cfg.get("max_rad_s", 50.0))
        self._gyro_max_jump_rad_s = float(filter_cfg.get("max_jump_rad_s", 20.0))
        self._gyro_max_age_s = float(filter_cfg.get("max_age_s", 0.2))
        self._gyro_temp_min_c = float(filter_cfg.get("temp_min_c", -50.0))
        self._gyro_temp_max_c = float(filter_cfg.get("temp_max_c", 150.0))
        self._max_dt_s = float(cfg.get("max_dt_s", 0.2))
        if self._max_dt_s <= 0.0:
            self._max_dt_s = 0.2

        yaw_buf_cfg = cfg.get("yaw_buffer", {}) if isinstance(cfg.get("yaw_buffer", {}), dict) else {}
        self._yaw_max_samples = max(64, int(yaw_buf_cfg.get("max_samples", 6000)))

        self._gyro = GyroState()
        self._gyro_lock = threading.Lock()
        self._yaw_samples: collections.deque[ImuYawSample] = collections.deque(maxlen=self._yaw_max_samples)
        self._yaw_lock = threading.Lock()

        self._last_stamp_s: float | None = None
        self._yaw_cum = 0.0

    def _convert_frame(self, gx: float, gy: float, gz: float) -> tuple[float, float, float]:
        # ROS IMU is typically in base_link (FLU). PX4 expects body FRD.
        frame = self._frame
        if frame in ("ros_flu", "ros_enu", "ros", "flu", "enu"):
            return gx, -gy, -gz
        if frame in ("px4_frd", "frd", "ned", "px4"):
            return gx, gy, gz
        return gx, gy, gz

    def _accept_gyro_sample(self, x: float, y: float, z: float) -> bool:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False
        if abs(x) > self._gyro_max_rad_s or abs(y) > self._gyro_max_rad_s or abs(z) > self._gyro_max_rad_s:
            return False
        if self._gyro.t_wall > 0.0:
            if (
                abs(x - self._gyro.x_rad_s) > self._gyro_max_jump_rad_s
                or abs(y - self._gyro.y_rad_s) > self._gyro_max_jump_rad_s
                or abs(z - self._gyro.z_rad_s) > self._gyro_max_jump_rad_s
            ):
                return False
        return True

    def _accept_temp(self, t: float) -> bool:
        if not math.isfinite(t):
            return False
        return self._gyro_temp_min_c <= t <= self._gyro_temp_max_c

    def _append_yaw_sample(self, t_imu_s: float, yaw_rad: float, t_wall: float) -> None:
        if not (math.isfinite(t_imu_s) and math.isfinite(yaw_rad)):
            return
        sample = ImuYawSample(t_imu_s=float(t_imu_s), yaw_rad=float(yaw_rad), t_wall=float(t_wall))
        with self._yaw_lock:
            if (not self._yaw_samples) or (sample.t_imu_s >= self._yaw_samples[-1].t_imu_s):
                self._yaw_samples.append(sample)
                return
            buf = list(self._yaw_samples)
            insert_at = len(buf)
            while insert_at > 0 and sample.t_imu_s < buf[insert_at - 1].t_imu_s:
                insert_at -= 1
            buf.insert(insert_at, sample)
            if len(buf) > self._yaw_max_samples:
                buf = buf[-self._yaw_max_samples :]
            self._yaw_samples = collections.deque(buf, maxlen=self._yaw_max_samples)

    def _ingest_sample(self, gx: float, gy: float, gz: float, stamp_s: float | None) -> None:
        t_wall = time.monotonic()
        gx, gy, gz = self._convert_frame(gx, gy, gz)
        with self._gyro_lock:
            if self._accept_gyro_sample(gx, gy, gz):
                self._gyro.x_rad_s = float(gx)
                self._gyro.y_rad_s = float(gy)
                self._gyro.z_rad_s = float(gz)
                self._gyro.t_wall = t_wall

        if stamp_s is None:
            return
        if self._last_stamp_s is None:
            self._last_stamp_s = float(stamp_s)
        else:
            dt = float(stamp_s) - float(self._last_stamp_s)
            if math.isfinite(dt) and dt > 0.0:
                if self._max_dt_s > 0.0:
                    dt = min(dt, self._max_dt_s)
                self._yaw_cum += float(gz) * dt
            if math.isfinite(float(stamp_s)):
                self._last_stamp_s = float(stamp_s)
        if math.isfinite(float(stamp_s)):
            self._append_yaw_sample(float(stamp_s), float(self._yaw_cum), t_wall)

    def read_gyro(self) -> GyroState:
        with self._gyro_lock:
            if self._gyro_max_age_s > 0.0 and self._gyro.t_wall > 0.0:
                if (time.monotonic() - self._gyro.t_wall) > self._gyro_max_age_s:
                    return GyroState()
            return GyroState(
                x_rad_s=self._gyro.x_rad_s,
                y_rad_s=self._gyro.y_rad_s,
                z_rad_s=self._gyro.z_rad_s,
                temperature_c=self._gyro.temperature_c,
                t_wall=self._gyro.t_wall,
            )

    def read_yaw_samples(self, max_samples: int = 0, max_age_s: float = 0.0) -> list[ImuYawSample]:
        with self._yaw_lock:
            out = list(self._yaw_samples)
        if max_samples > 0 and len(out) > max_samples:
            out = out[-int(max_samples) :]
        if max_age_s > 0.0 and out:
            newest_t = out[-1].t_imu_s
            cutoff = newest_t - float(max_age_s)
            out = [s for s in out if s.t_imu_s >= cutoff]
        return out


class Ros1ImuSubscriber(_BaseImuSubscriber):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self._topic = str(cfg.get("topic", "/mavros/imu/data_raw"))
        self._node_name = str(cfg.get("ros_node_name", "px4flow_rpi_imu"))
        self._stop = threading.Event()
        self._sub = None

        try:
            import rospy  # type: ignore
            from sensor_msgs.msg import Imu  # type: ignore
        except Exception as exc:
            raise RuntimeError("ros1 imu backend requires rospy and sensor_msgs (source ROS 1 environment first)") from exc

        self._rospy = rospy
        self._owns_rospy = False
        try:
            if not rospy.core.is_initialized():  # type: ignore[attr-defined]
                rospy.init_node(self._node_name, anonymous=True, disable_signals=True)
                self._owns_rospy = True
        except Exception:
            try:
                rospy.get_name()
            except Exception:
                rospy.init_node(self._node_name, anonymous=True, disable_signals=True)
                self._owns_rospy = True

        ros_queue = max(1, int(cfg.get("ros_queue_size", 50)))

        def _on_imu(msg: Imu) -> None:
            try:
                gx = float(msg.angular_velocity.x)
                gy = float(msg.angular_velocity.y)
                gz = float(msg.angular_velocity.z)
            except Exception:
                return
            stamp_s = None
            try:
                stamp = msg.header.stamp
                if hasattr(stamp, "to_sec"):
                    stamp_s = float(stamp.to_sec())
                else:
                    sec = getattr(stamp, "secs", None)
                    nsec = getattr(stamp, "nsecs", None)
                    if sec is not None and nsec is not None:
                        stamp_s = float(sec) + float(nsec) * 1e-9
            except Exception:
                stamp_s = None
            self._ingest_sample(gx, gy, gz, stamp_s)

        self._sub = rospy.Subscriber(self._topic, Imu, _on_imu, queue_size=ros_queue)

    def close(self) -> None:
        self._stop.set()
        if self._sub is not None:
            try:
                self._sub.unregister()
            except Exception:
                pass
        if self._owns_rospy:
            try:
                self._rospy.signal_shutdown("px4flow_rpi imu shutdown")
            except Exception:
                pass


class Ros2ImuSubscriber(_BaseImuSubscriber):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self._topic = str(cfg.get("topic", "/mavros/imu/data_raw"))
        self._node_name = str(cfg.get("ros_node_name", "px4flow_rpi_imu"))
        self._spin_timeout_s = max(0.001, float(cfg.get("ros_spin_timeout_s", 0.05)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        try:
            import rclpy  # type: ignore
            from rclpy.executors import SingleThreadedExecutor  # type: ignore
            from rclpy.node import Node  # type: ignore
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # type: ignore
            from sensor_msgs.msg import Imu  # type: ignore
        except Exception as exc:
            raise RuntimeError("ros2 imu backend requires rclpy and sensor_msgs (source ROS 2 environment first)") from exc

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

        def _on_imu(msg: Imu) -> None:
            try:
                gx = float(msg.angular_velocity.x)
                gy = float(msg.angular_velocity.y)
                gz = float(msg.angular_velocity.z)
            except Exception:
                return
            stamp_s = None
            try:
                stamp = msg.header.stamp
                sec = getattr(stamp, "sec", None)
                nsec = getattr(stamp, "nanosec", None)
                if sec is None:
                    sec = getattr(stamp, "secs", None)
                if nsec is None:
                    nsec = getattr(stamp, "nsecs", None)
                if sec is not None and nsec is not None:
                    stamp_s = float(sec) + float(nsec) * 1e-9
            except Exception:
                stamp_s = None
            self._ingest_sample(gx, gy, gz, stamp_s)

        self._sub = self._node.create_subscription(Imu, self._topic, _on_imu, qos)
        self._thread = threading.Thread(target=self._spin_loop, name="ros2-imu", daemon=True)
        self._thread.start()

    def _spin_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=self._spin_timeout_s)
            except Exception:
                time.sleep(0.01)

    def close(self) -> None:
        self._stop.set()
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
