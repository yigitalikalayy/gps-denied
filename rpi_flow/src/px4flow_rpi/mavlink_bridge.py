from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import serial  # type: ignore

from .mavlink_messages import (
    Attitude,
    AttitudeQuaternion,
    HighresImu,
    ScaledImu,
    pack_command_long,
    pack_distance_sensor,
    pack_optical_flow_rad,
    pack_request_data_stream,
)
from .mavlink_v1 import Mavlink1Parser, mavlink1_pack


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


class MavlinkBridge:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._port = str(cfg.get("port", "/dev/ttyAMA1"))
        self._baud = int(cfg.get("baudrate", 921600))
        self._sysid = int(cfg.get("sysid", 42))
        self._compid = int(cfg.get("compid", 197))
        self._target_sysid = int(cfg.get("target_sysid", 1))
        self._target_compid = int(cfg.get("target_compid", 1))

        self._seq = 0
        self._parser = Mavlink1Parser()
        self._ser = serial.Serial(self._port, self._baud, timeout=0.01)

        self._gyro = GyroState()
        self._gyro_lock = threading.Lock()
        yaw_buf_cfg = cfg.get("yaw_buffer", {}) if isinstance(cfg.get("yaw_buffer", {}), dict) else {}
        self._yaw_max_samples = max(64, int(yaw_buf_cfg.get("max_samples", 6000)))
        self._yaw_samples: collections.deque[ImuYawSample] = collections.deque(maxlen=self._yaw_max_samples)
        self._yaw_lock = threading.Lock()

        self._imu_req_cfg = cfg.get("request_imu_stream", {}) if isinstance(cfg.get("request_imu_stream", {}), dict) else {}
        imu_filter_cfg = cfg.get("imu_filter", {}) if isinstance(cfg.get("imu_filter", {}), dict) else {}
        self._gyro_max_rad_s = float(imu_filter_cfg.get("max_rad_s", 50.0))
        self._gyro_max_jump_rad_s = float(imu_filter_cfg.get("max_jump_rad_s", 20.0))
        self._gyro_max_age_s = float(imu_filter_cfg.get("max_age_s", 0.2))
        self._gyro_temp_min_c = float(imu_filter_cfg.get("temp_min_c", -50.0))
        self._gyro_temp_max_c = float(imu_filter_cfg.get("temp_max_c", 150.0))

        self._stop = False
        self._rx_thread = threading.Thread(target=self._rx_loop, name="mavlink-rx", daemon=True)
        self._rx_thread.start()
        if bool(self._imu_req_cfg.get("enabled", False)):
            self.request_imu_stream()

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

    @staticmethod
    def _wrap_pi(v: float) -> float:
        return math.atan2(math.sin(v), math.cos(v))

    def _append_yaw_sample(self, t_imu_s: float, yaw_rad: float, t_wall: float) -> None:
        if not (math.isfinite(t_imu_s) and math.isfinite(yaw_rad)):
            return
        sample = ImuYawSample(t_imu_s=float(t_imu_s), yaw_rad=self._wrap_pi(float(yaw_rad)), t_wall=float(t_wall))
        with self._yaw_lock:
            if (not self._yaw_samples) or (sample.t_imu_s >= self._yaw_samples[-1].t_imu_s):
                self._yaw_samples.append(sample)
                return
            # UART jitter can deliver packets out of order. Keep queue timestamp-sorted.
            buf = list(self._yaw_samples)
            insert_at = len(buf)
            while insert_at > 0 and sample.t_imu_s < buf[insert_at - 1].t_imu_s:
                insert_at -= 1
            buf.insert(insert_at, sample)
            if len(buf) > self._yaw_max_samples:
                buf = buf[-self._yaw_max_samples :]
            self._yaw_samples = collections.deque(buf, maxlen=self._yaw_max_samples)

    def close(self) -> None:
        self._stop = True
        try:
            self._rx_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass

    def _rx_loop(self) -> None:
        while not self._stop:
            try:
                data = self._ser.read(512)
            except Exception:
                time.sleep(0.01)
                continue
            if not data:
                continue

            for f in self._parser.feed(data):
                try:
                    if f.msgid == 105 and len(f.payload) >= 62:
                        imu = HighresImu.decode(f.payload)
                        with self._gyro_lock:
                            gx = float(imu.xgyro)
                            gy = float(imu.ygyro)
                            gz = float(imu.zgyro)
                            if self._accept_gyro_sample(gx, gy, gz):
                                self._gyro.x_rad_s = gx
                                self._gyro.y_rad_s = gy
                                self._gyro.z_rad_s = gz
                                if self._accept_temp(float(imu.temperature)):
                                    self._gyro.temperature_c = float(imu.temperature)
                                self._gyro.t_wall = f.timestamp
                    elif f.msgid == 26 and len(f.payload) >= 22:
                        imu = ScaledImu.decode(f.payload)
                        with self._gyro_lock:
                            gx = float(imu.xgyro_mrad_s) / 1000.0
                            gy = float(imu.ygyro_mrad_s) / 1000.0
                            gz = float(imu.zgyro_mrad_s) / 1000.0
                            if self._accept_gyro_sample(gx, gy, gz):
                                self._gyro.x_rad_s = gx
                                self._gyro.y_rad_s = gy
                                self._gyro.z_rad_s = gz
                                self._gyro.t_wall = f.timestamp
                    elif f.msgid == 30 and len(f.payload) >= 16:
                        att = Attitude.decode(f.payload)
                        self._append_yaw_sample(float(att.time_boot_ms) * 1e-3, float(att.yaw), f.timestamp)
                    elif f.msgid == 31 and len(f.payload) >= 20:
                        attq = AttitudeQuaternion.decode(f.payload)
                        self._append_yaw_sample(float(attq.time_boot_ms) * 1e-3, float(attq.yaw_rad()), f.timestamp)
                except Exception:
                    continue

    def _send(self, msgid: int, payload: bytes, crc_extra: int) -> None:
        pkt = mavlink1_pack(msgid, payload, sysid=self._sysid, compid=self._compid, seq=self._seq, crc_extra=crc_extra)
        self._seq = (self._seq + 1) & 0xFF
        self._ser.write(pkt)

    def request_imu_stream(self) -> None:
        stream_id = int(self._imu_req_cfg.get("stream_id", 0))
        rate_hz = int(self._imu_req_cfg.get("rate_hz", 200))
        method = str(self._imu_req_cfg.get("method", "auto")).strip().lower()
        msg_ids = self._imu_req_cfg.get("message_ids", [105, 26])
        if isinstance(msg_ids, (int, float, str)):
            msg_ids = [msg_ids]

        if method in ("auto", "command_long", "set_message_interval"):
            if rate_hz > 0:
                interval_us = float(1_000_000.0 / max(1, rate_hz))
            else:
                interval_us = -1.0
            for mid in msg_ids:
                try:
                    mid_i = int(mid)
                except Exception:
                    continue
                msgid, payload, crc_extra = pack_command_long(
                    target_sysid=self._target_sysid,
                    target_compid=self._target_compid,
                    command=511,  # MAV_CMD_SET_MESSAGE_INTERVAL
                    confirmation=0,
                    param1=float(mid_i),
                    param2=float(interval_us),
                    param3=0.0,
                    param4=0.0,
                    param5=0.0,
                    param6=0.0,
                    param7=0.0,
                )
                self._send(msgid, payload, crc_extra)

        if method in ("auto", "request_data_stream", "stream"):
            msgid, payload, crc_extra = pack_request_data_stream(
                target_sysid=self._target_sysid,
                target_compid=self._target_compid,
                stream_id=stream_id,
                rate_hz=rate_hz,
                start_stop=1,
            )
            self._send(msgid, payload, crc_extra)

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

    def send_optical_flow_rad(
        self,
        *,
        time_usec: int,
        sensor_id: int,
        integration_time_us: int,
        integrated_x: float,
        integrated_y: float,
        integrated_xgyro: float,
        integrated_ygyro: float,
        integrated_zgyro: float,
        temperature: float,
        quality: int,
        time_delta_distance_us: int,
        distance_m: float,
    ) -> None:
        msgid, payload, crc_extra = pack_optical_flow_rad(
            time_usec=time_usec,
            sensor_id=sensor_id,
            integration_time_us=integration_time_us,
            integrated_x=integrated_x,
            integrated_y=integrated_y,
            integrated_xgyro=integrated_xgyro,
            integrated_ygyro=integrated_ygyro,
            integrated_zgyro=integrated_zgyro,
            temperature=temperature,
            quality=quality,
            time_delta_distance_us=time_delta_distance_us,
            distance_m=distance_m,
        )
        self._send(msgid, payload, crc_extra)

    def send_distance_sensor(
        self,
        *,
        time_boot_ms: int,
        min_distance_m: float,
        max_distance_m: float,
        current_distance_m: float,
        sensor_type: int = 0,
        sensor_id: int = 0,
        orientation: int = 25,
        covariance: int = 255,
    ) -> None:
        if not math.isfinite(current_distance_m) or current_distance_m <= 0.0:
            return
        if not math.isfinite(min_distance_m) or min_distance_m < 0.0:
            min_distance_m = 0.0
        if not math.isfinite(max_distance_m) or max_distance_m <= 0.0:
            max_distance_m = max(min_distance_m, current_distance_m)

        def _to_cm(v_m: float) -> int:
            return int(round(max(0.0, v_m) * 100.0))

        msgid, payload, crc_extra = pack_distance_sensor(
            time_boot_ms=time_boot_ms,
            min_distance_cm=_to_cm(min_distance_m),
            max_distance_cm=_to_cm(max_distance_m),
            current_distance_cm=_to_cm(current_distance_m),
            sensor_type=sensor_type,
            sensor_id=sensor_id,
            orientation=orientation,
            covariance=covariance,
        )
        self._send(msgid, payload, crc_extra)


def monotonic_to_time_usec(t0_monotonic: float, t0_time: float, t_monotonic: float) -> int:
    return int((t0_time + (t_monotonic - t0_monotonic)) * 1_000_000)
