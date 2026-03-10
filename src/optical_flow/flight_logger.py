from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class FlightLogRow:
    timestamp_s: float
    frame_id: int
    feature_count: int
    tracked_feature_count: int
    tracking_loss_rate: float
    flow_px_x: float
    flow_px_y: float
    flow_rad_x: float
    flow_rad_y: float
    flow_quality: int
    gyro_x: float
    gyro_y: float
    gyro_z: float
    integrated_xgyro: float
    integrated_ygyro: float
    integrated_zgyro: float
    raw_range: float | None
    filtered_range: float | None
    range_valid: bool
    range_delay_us: int | None


class FlightLogger:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._enabled = bool(cfg.get("enabled", False))
        self._file_prefix = str(cfg.get("file_prefix", "flight")).strip() or "flight"
        self._base_dir = str(cfg.get("dir", "logs/flights")).strip() or "logs/flights"
        self._flush_interval_s = float(cfg.get("flush_interval_s", 1.0))
        self._last_flush = time.monotonic()
        self._writer = None
        self._fp = None
        if not self._enabled:
            return
        os.makedirs(self._base_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_num = 1
        for name in os.listdir(self._base_dir):
            if not name.startswith(f"{self._file_prefix}_"):
                continue
            parts = name.split("_")
            if len(parts) < 2:
                continue
            try:
                num = int(parts[1])
            except Exception:
                continue
            log_num = max(log_num, num + 1)
        path = os.path.join(self._base_dir, f"{self._file_prefix}_{log_num:04d}_{timestamp}.csv")
        self._fp = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fp, fieldnames=self._fieldnames())
        self._writer.writeheader()
        self._fp.flush()

    @staticmethod
    def _fieldnames() -> list[str]:
        return [
            "timestamp_s",
            "frame_id",
            "feature_count",
            "tracked_feature_count",
            "tracking_loss_rate",
            "flow_px_x",
            "flow_px_y",
            "flow_rad_x",
            "flow_rad_y",
            "flow_quality",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "integrated_xgyro",
            "integrated_ygyro",
            "integrated_zgyro",
            "raw_range",
            "filtered_range",
            "range_valid",
            "range_delay_us",
        ]

    def log(self, **kwargs: Any) -> None:
        if not self._enabled or self._writer is None or self._fp is None:
            return
        row = FlightLogRow(**kwargs)
        self._writer.writerow(row.__dict__)
        if self._flush_interval_s > 0.0:
            now = time.monotonic()
            if (now - self._last_flush) >= self._flush_interval_s:
                self._fp.flush()
                self._last_flush = now

    def close(self) -> None:
        if not self._enabled:
            return
        if self._fp is not None:
            try:
                self._fp.flush()
            except Exception:
                pass
            try:
                self._fp.close()
            except Exception:
                pass
        self._fp = None
        self._writer = None
