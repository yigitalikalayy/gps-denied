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
        self._ascii_cfg = cfg.get("ascii", {}) if isinstance(cfg.get("ascii", {}), dict) else {}
        self._init_cmd = str(self._ascii_cfg.get("init_command", "?P"))
        self._dist_cmd = str(self._ascii_cfg.get("distance_command", "?LDF,0"))
        self._max_read_len = int(self._ascii_cfg.get("max_read_len", 32))

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
        resp = self._exchange(self._dist_cmd).lower()

        # Expected: "ldf,0:56.98" (units: meters) with CR/LF.
        m = re.search(r":\s*([-+]?\d+(?:\.\d+)?)", resp)
        if not m:
            raise RuntimeError(f"lw20 parse failed: {resp!r}")
        distance_m = float(m.group(1))
        return LidarSample(distance_m=distance_m, t_monotonic=t)

    def close(self) -> None:
        try:
            self._bus_handle.close()
        except Exception:
            pass
