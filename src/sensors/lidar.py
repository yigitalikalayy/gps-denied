from __future__ import annotations

import re
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
