from __future__ import annotations

import math
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


@dataclass
class Attitude:
    time_boot_ms: int
    roll: float
    pitch: float
    yaw: float
    rollspeed: float
    pitchspeed: float
    yawspeed: float

    @staticmethod
    def decode(payload: bytes) -> "Attitude":
        # MAVLink ATTITUDE (id=30):
        # time_boot_ms (u32), roll/pitch/yaw (float*3), roll/pitch/yaw speed (float*3)
        if len(payload) < 28:
            raise ValueError("ATTITUDE payload too short")
        time_boot_ms = struct.unpack_from("<I", payload, 0)[0]
        roll, pitch, yaw = struct.unpack_from("<fff", payload, 4)
        rollspeed, pitchspeed, yawspeed = struct.unpack_from("<fff", payload, 16)
        return Attitude(
            time_boot_ms=time_boot_ms,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            rollspeed=rollspeed,
            pitchspeed=pitchspeed,
            yawspeed=yawspeed,
        )


@dataclass
class AttitudeQuaternion:
    time_boot_ms: int
    q1: float
    q2: float
    q3: float
    q4: float

    @staticmethod
    def decode(payload: bytes) -> "AttitudeQuaternion":
        # MAVLink ATTITUDE_QUATERNION (id=31), first fields:
        # time_boot_ms (u32), q1,q2,q3,q4 (float*4)
        if len(payload) < 20:
            raise ValueError("ATTITUDE_QUATERNION payload too short")
        time_boot_ms = struct.unpack_from("<I", payload, 0)[0]
        q1, q2, q3, q4 = struct.unpack_from("<ffff", payload, 4)
        return AttitudeQuaternion(time_boot_ms=time_boot_ms, q1=q1, q2=q2, q3=q3, q4=q4)

    def yaw_rad(self) -> float:
        # Autopilot quaternion convention is w,x,y,z => q1,q2,q3,q4.
        siny_cosp = 2.0 * (self.q1 * self.q4 + self.q2 * self.q3)
        cosy_cosp = 1.0 - 2.0 * (self.q3 * self.q3 + self.q4 * self.q4)
        return math.atan2(siny_cosp, cosy_cosp)


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


def pack_command_long(
    *,
    target_sysid: int,
    target_compid: int,
    command: int,
    confirmation: int = 0,
    param1: float = 0.0,
    param2: float = 0.0,
    param3: float = 0.0,
    param4: float = 0.0,
    param5: float = 0.0,
    param6: float = 0.0,
    param7: float = 0.0,
) -> tuple[int, bytes, int]:
    """
    COMMAND_LONG (msgid=76, crc_extra=152)
    Payload: param1..param7 (float*7), command (u16),
             target_system (u8), target_component (u8), confirmation (u8)
    """
    msgid = 76
    crc_extra = 152
    payload = struct.pack(
        "<fffffffHBBB",
        float(param1),
        float(param2),
        float(param3),
        float(param4),
        float(param5),
        float(param6),
        float(param7),
        int(command) & 0xFFFF,
        int(target_sysid) & 0xFF,
        int(target_compid) & 0xFF,
        int(confirmation) & 0xFF,
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

    def _finite(v: float, default: float = 0.0) -> float:
        try:
            if math.isfinite(v):
                return float(v)
        except Exception:
            pass
        return default

    quality_u8 = max(0, min(255, int(_finite(float(quality), 0.0))))
    # MAVLink OPTICAL_FLOW_RAD temperature is in cdegC (degC * 100).
    temp_cdeg = _finite(float(temperature), 0.0) * 100.0
    temp_i16 = max(-32768, min(32767, int(round(temp_cdeg))))

    payload = struct.pack(
        "<QIfffffIfhBB",
        time_usec & 0xFFFFFFFFFFFFFFFF,
        integration_time_us & 0xFFFFFFFF,
        _finite(integrated_x),
        _finite(integrated_y),
        _finite(integrated_xgyro),
        _finite(integrated_ygyro),
        _finite(integrated_zgyro),
        time_delta_distance_us & 0xFFFFFFFF,
        _finite(distance_m),
        temp_i16,
        sensor_id & 0xFF,
        quality_u8,
    )
    return msgid, payload, crc_extra


def pack_optical_flow(
    time_usec: int,
    sensor_id: int,
    flow_x: int,
    flow_y: int,
    flow_comp_m_x: float,
    flow_comp_m_y: float,
    quality: int,
    ground_distance: float,
) -> tuple[int, bytes, int]:
    """
    OPTICAL_FLOW (msgid=100, crc_extra=175)
    Base (non-extension) fields only; MAVLink1 frame.
    """
    msgid = 100
    crc_extra = 175

    def _finite(v: float, default: float = 0.0) -> float:
        try:
            if math.isfinite(v):
                return float(v)
        except Exception:
            pass
        return default

    def _i16(v: int) -> int:
        return max(-32768, min(32767, int(v)))

    quality_u8 = max(0, min(255, int(_finite(float(quality), 0.0))))

    payload = struct.pack(
        "<QfffhhBB",
        time_usec & 0xFFFFFFFFFFFFFFFF,
        _finite(flow_comp_m_x),
        _finite(flow_comp_m_y),
        _finite(ground_distance),
        _i16(flow_x),
        _i16(flow_y),
        sensor_id & 0xFF,
        quality_u8,
    )
    return msgid, payload, crc_extra


def pack_hil_optical_flow(
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
    HIL_OPTICAL_FLOW (msgid=114, crc_extra=237)
    """
    msgid = 114
    crc_extra = 237

    def _finite(v: float, default: float = 0.0) -> float:
        try:
            if math.isfinite(v):
                return float(v)
        except Exception:
            pass
        return default

    quality_u8 = max(0, min(255, int(_finite(float(quality), 0.0))))
    # MAVLink HIL_OPTICAL_FLOW temperature is in cdegC (degC * 100).
    temp_cdeg = _finite(float(temperature), 0.0) * 100.0
    temp_i16 = max(-32768, min(32767, int(round(temp_cdeg))))

    payload = struct.pack(
        "<QIfffffIfhBB",
        time_usec & 0xFFFFFFFFFFFFFFFF,
        integration_time_us & 0xFFFFFFFF,
        _finite(integrated_x),
        _finite(integrated_y),
        _finite(integrated_xgyro),
        _finite(integrated_ygyro),
        _finite(integrated_zgyro),
        time_delta_distance_us & 0xFFFFFFFF,
        _finite(distance_m),
        temp_i16,
        sensor_id & 0xFF,
        quality_u8,
    )
    return msgid, payload, crc_extra


def pack_distance_sensor(
    time_boot_ms: int,
    min_distance_cm: int,
    max_distance_cm: int,
    current_distance_cm: int,
    sensor_type: int,
    sensor_id: int,
    orientation: int,
    covariance: int,
) -> tuple[int, bytes, int]:
    """
    DISTANCE_SENSOR (msgid=132, crc_extra=85)
    Payload: time_boot_ms (u32), min_distance (u16), max_distance (u16),
             current_distance (u16), type (u8), id (u8), orientation (u8), covariance (u8)
    """
    msgid = 132
    crc_extra = 85

    def _u16(v: int) -> int:
        return max(0, min(65535, int(v)))

    payload = struct.pack(
        "<IHHHBBBB",
        int(time_boot_ms) & 0xFFFFFFFF,
        _u16(min_distance_cm),
        _u16(max_distance_cm),
        _u16(current_distance_cm),
        int(sensor_type) & 0xFF,
        int(sensor_id) & 0xFF,
        int(orientation) & 0xFF,
        int(covariance) & 0xFF,
    )
    return msgid, payload, crc_extra
