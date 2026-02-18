from __future__ import annotations

import collections
import re
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class LidarSample:
    distance_m: float
    t_monotonic: float


class LightwareLw20AsciiI2C:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._bus = int(cfg.get("i2c_bus", 1))
        self._addr = int(cfg.get("i2c_address", 0x66))
        self._scale = float(cfg.get("scale", 1.0))
        self.last_raw = ""
        self._ascii_cfg = cfg.get("ascii", {}) if isinstance(cfg.get("ascii", {}), dict) else {}
        self._init_cmd = str(self._ascii_cfg.get("init_command", "?P"))
        self._dist_cmd = str(self._ascii_cfg.get("distance_command", "?LDF,0"))
        self._max_read_len = int(self._ascii_cfg.get("max_read_len", 32))
        self._value_index = int(self._ascii_cfg.get("value_index", 0))

        try:
            from smbus2 import SMBus, i2c_msg  # type: ignore
        except Exception:
            raise RuntimeError("LW20 ASCII I2C backend requires smbus2")

        self._SMBus = SMBus
        self._i2c_msg = i2c_msg
        self._bus_handle = self._SMBus(self._bus)
        self._prime()

    def _prime(self) -> None:
        # Lightware docs: first command after power-up may not return a response.
        try:
            _ = self._exchange(self._init_cmd)
            _ = self._exchange(self._init_cmd)
        except Exception:
            pass

    def _exchange(self, cmd: str) -> str:
        payload = (cmd.strip() + "\r\n").encode("ascii", errors="strict")

        # Prefer a repeated-start (write+read) transaction.
        try:
            w = self._i2c_msg.write(self._addr, payload)
            r = self._i2c_msg.read(self._addr, self._max_read_len)
            self._bus_handle.i2c_rdwr(w, r)
            raw = bytes(list(r))
        except Exception:
            # Fallback: separate write then read.
            w = self._i2c_msg.write(self._addr, payload)
            self._bus_handle.i2c_rdwr(w)
            time.sleep(0.002)
            r = self._i2c_msg.read(self._addr, self._max_read_len)
            self._bus_handle.i2c_rdwr(r)
            raw = bytes(list(r))

        text = raw.decode("ascii", errors="ignore").replace("\x00", "").strip()
        return text

    def read(self) -> LidarSample:
        t = time.monotonic()
        raw = self._exchange(self._dist_cmd)
        self.last_raw = raw
        resp = raw.lower()

        # Expected examples: "ldf,0:56.98" or "0:0.09 0,16".
        segment = resp.split(":", 1)[1] if ":" in resp else resp
        matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?", segment)
        if not matches:
            raise RuntimeError(f"lw20 parse failed: {resp!r}")
        idx = max(0, min(self._value_index, len(matches) - 1))
        token = matches[idx].replace(",", ".")
        distance_m = float(token) * self._scale
        return LidarSample(distance_m=distance_m, t_monotonic=t)

    def close(self) -> None:
        try:
            self._bus_handle.close()
        except Exception:
            pass


class LightwareLw20BinaryI2C:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._bus = int(cfg.get("i2c_bus", 1))
        self._addr = int(cfg.get("i2c_address", 0x66))
        self._reg = int(cfg.get("register", 0x00))

        try:
            from smbus2 import SMBus, i2c_msg  # type: ignore
        except Exception:
            raise RuntimeError("LW20 binary I2C backend requires smbus2")

        self._SMBus = SMBus
        self._i2c_msg = i2c_msg
        self._bus_handle = self._SMBus(self._bus)

    def _read_raw_cm(self) -> int:
        reg = bytes([self._reg & 0xFF])
        try:
            w = self._i2c_msg.write(self._addr, reg)
            r = self._i2c_msg.read(self._addr, 2)
            self._bus_handle.i2c_rdwr(w, r)
            raw = bytes(list(r))
        except Exception:
            w = self._i2c_msg.write(self._addr, reg)
            self._bus_handle.i2c_rdwr(w)
            time.sleep(0.002)
            r = self._i2c_msg.read(self._addr, 2)
            self._bus_handle.i2c_rdwr(r)
            raw = bytes(list(r))
        if len(raw) != 2:
            raise RuntimeError(f"lw20 binary read short: {raw!r}")
        return (int(raw[0]) << 8) | int(raw[1])

    def read(self) -> LidarSample:
        t = time.monotonic()
        raw_cm = self._read_raw_cm()
        if raw_cm == 0 or raw_cm == 0xFFFF:
            raise RuntimeError(f"lw20 invalid range: 0x{raw_cm:04x}")
        distance_m = float(raw_cm) / 100.0
        return LidarSample(distance_m=distance_m, t_monotonic=t)

    def close(self) -> None:
        try:
            self._bus_handle.close()
        except Exception:
            pass


class Ros1RangeSubscriber:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._topic = str(cfg.get("topic", "/distance_sensor"))
        self._node_name = str(cfg.get("ros_node_name", "px4flow_rpi_lidar"))
        self._timeout_s = max(0.05, float(cfg.get("ros_timeout_s", 1.0)))
        self._queue: collections.deque[LidarSample] = collections.deque(
            maxlen=max(1, int(cfg.get("queue_size", 5)))
        )
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._sub = None

        try:
            import rospy  # type: ignore
            from sensor_msgs.msg import Range  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "ros1 lidar backend requires rospy and sensor_msgs (source ROS 1 environment first)"
            ) from exc

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

        ros_queue = max(1, int(cfg.get("ros_queue_size", 10)))

        def _on_range(msg: Range) -> None:
            try:
                d = float(msg.range)
            except Exception:
                return
            if not (d > 0.0):
                return
            sample = LidarSample(distance_m=d, t_monotonic=time.monotonic())
            with self._cv:
                self._queue.append(sample)
                self._cv.notify_all()

        self._sub = rospy.Subscriber(self._topic, Range, _on_range, queue_size=ros_queue)

    def read(self, timeout_s: float | None = None) -> LidarSample:
        timeout = self._timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout
        with self._cv:
            while (not self._queue) and (not self._stop.is_set()):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._cv.wait(timeout=remaining)
            if not self._queue:
                raise RuntimeError(f"ros1 range topic timeout: {self._topic}")
            sample = self._queue.pop()
            self._queue.clear()
            return sample

    def close(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._sub is not None:
            try:
                self._sub.unregister()
            except Exception:
                pass
        if self._owns_rospy:
            try:
                self._rospy.signal_shutdown("px4flow_rpi lidar shutdown")
            except Exception:
                pass
