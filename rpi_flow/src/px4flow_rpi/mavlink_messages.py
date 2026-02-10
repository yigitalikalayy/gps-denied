from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class HighresImu:
    time_usec: int
    xgyro: float
    ygyro: float
    zgyro: float
    temperature: float

    @staticmethod
    def decode(payload: bytes) -> "HighresImu":
        # MAVLink HIGHRES_IMU (id=105) payload layout (little-endian)
        # time_usec (uint64) at 0
        # xacc,yacc,zacc (float*3) at 8
        # xgyro,ygyro,zgyro (float*3) at 20
        # temperature (float) at 56
        if len(payload) < 62:
            raise ValueError("HIGHRES_IMU payload too short")
        time_usec = struct.unpack_from("<Q", payload, 0)[0]
        xgyro, ygyro, zgyro = struct.unpack_from("<fff", payload, 20)
        temperature = struct.unpack_from("<f", payload, 56)[0]
        return HighresImu(time_usec=time_usec, xgyro=xgyro, ygyro=ygyro, zgyro=zgyro, temperature=temperature)


@dataclass
class ScaledImu:
    time_boot_ms: int
    xgyro_mrad_s: int
    ygyro_mrad_s: int
    zgyro_mrad_s: int

    @staticmethod
    def decode(payload: bytes) -> "ScaledImu":
        if len(payload) < 22:
            raise ValueError("SCALED_IMU payload too short")
        time_boot_ms = struct.unpack_from("<I", payload, 0)[0]
        xgyro, ygyro, zgyro = struct.unpack_from("<hhh", payload, 10)
        return ScaledImu(time_boot_ms=time_boot_ms, xgyro_mrad_s=xgyro, ygyro_mrad_s=ygyro, zgyro_mrad_s=zgyro)


def pack_request_data_stream(
    target_sysid: int,
    target_compid: int,
    stream_id: int,
    rate_hz: int,
    start_stop: int,
) -> tuple[int, bytes, int]:
    """
    REQUEST_DATA_STREAM (msgid=66, crc_extra=148)
    Payload: target_system (u8), target_component (u8), req_stream_id (u8),
             req_message_rate (u16), start_stop (u8)
    """
    msgid = 66
    crc_extra = 148
    payload = struct.pack(
        "<BBBHB",
        target_sysid & 0xFF,
        target_compid & 0xFF,
        stream_id & 0xFF,
        rate_hz & 0xFFFF,
        start_stop & 0xFF,
    )
    return msgid, payload, crc_extra


def pack_optical_flow_rad(
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
) -> tuple[int, bytes, int]:
    """
    OPTICAL_FLOW_RAD (msgid=106, crc_extra=138)
    """
    msgid = 106
    crc_extra = 138

    quality_u8 = max(0, min(255, int(quality)))
    temp_i16 = max(-32768, min(32767, int(round(temperature))))

    payload = struct.pack(
        "<QIfffffIfhBB",
        time_usec & 0xFFFFFFFFFFFFFFFF,
        integration_time_us & 0xFFFFFFFF,
        float(integrated_x),
        float(integrated_y),
        float(integrated_xgyro),
        float(integrated_ygyro),
        float(integrated_zgyro),
        time_delta_distance_us & 0xFFFFFFFF,
        float(distance_m),
        temp_i16,
        sensor_id & 0xFF,
        quality_u8,
    )
    return msgid, payload, crc_extra
