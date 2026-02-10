from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import serial  # type: ignore

from .mavlink_messages import HighresImu, ScaledImu, pack_optical_flow_rad, pack_request_data_stream
from .mavlink_v1 import Mavlink1Parser, mavlink1_pack


@dataclass
class GyroState:
    x_rad_s: float = 0.0
    y_rad_s: float = 0.0
    z_rad_s: float = 0.0
    temperature_c: float = 0.0
    t_wall: float = 0.0


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
        self._stop = False
        self._rx_thread = threading.Thread(target=self._rx_loop, name="mavlink-rx", daemon=True)
        self._rx_thread.start()

        self._imu_req_cfg = cfg.get("request_imu_stream", {}) if isinstance(cfg.get("request_imu_stream", {}), dict) else {}
        if bool(self._imu_req_cfg.get("enabled", False)):
            self.request_imu_stream()

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
                            self._gyro.x_rad_s = float(imu.xgyro)
                            self._gyro.y_rad_s = float(imu.ygyro)
                            self._gyro.z_rad_s = float(imu.zgyro)
                            self._gyro.temperature_c = float(imu.temperature)
                            self._gyro.t_wall = f.timestamp
                    elif f.msgid == 26 and len(f.payload) >= 22:
                        imu = ScaledImu.decode(f.payload)
                        with self._gyro_lock:
                            self._gyro.x_rad_s = float(imu.xgyro_mrad_s) / 1000.0
                            self._gyro.y_rad_s = float(imu.ygyro_mrad_s) / 1000.0
                            self._gyro.z_rad_s = float(imu.zgyro_mrad_s) / 1000.0
                            self._gyro.t_wall = f.timestamp
                except Exception:
                    continue

    def _send(self, msgid: int, payload: bytes, crc_extra: int) -> None:
        pkt = mavlink1_pack(msgid, payload, sysid=self._sysid, compid=self._compid, seq=self._seq, crc_extra=crc_extra)
        self._seq = (self._seq + 1) & 0xFF
        self._ser.write(pkt)

    def request_imu_stream(self) -> None:
        stream_id = int(self._imu_req_cfg.get("stream_id", 0))
        rate_hz = int(self._imu_req_cfg.get("rate_hz", 200))
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
            return GyroState(
                x_rad_s=self._gyro.x_rad_s,
                y_rad_s=self._gyro.y_rad_s,
                z_rad_s=self._gyro.z_rad_s,
                temperature_c=self._gyro.temperature_c,
                t_wall=self._gyro.t_wall,
            )

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


def monotonic_to_time_usec(t0_monotonic: float, t0_time: float, t_monotonic: float) -> int:
    return int((t0_time + (t_monotonic - t0_monotonic)) * 1_000_000)

