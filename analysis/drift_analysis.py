from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class DriftStats:
    duration_s: float
    total_distance_m: float
    path_length_m: float
    drift_per_minute_m: float
    drift_per_meter: float


def _iter_rows(paths: Iterable[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def _as_float(row, key, default=None):
    try:
        v = row.get(key)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def compute_drift(paths: List[str]) -> DriftStats:
    t0 = None
    last_t = None
    pos_x = 0.0
    pos_y = 0.0
    path_length = 0.0

    for row in _iter_rows(paths):
        t = _as_float(row, "timestamp_s")
        flow_rad_x = _as_float(row, "flow_rad_x")
        flow_rad_y = _as_float(row, "flow_rad_y")
        rng = _as_float(row, "filtered_range")
        if t is None or flow_rad_x is None or flow_rad_y is None or rng is None:
            continue
        if t0 is None:
            t0 = t
            last_t = t
            continue
        dt = max(1e-6, t - (last_t or t))
        last_t = t
        dx = flow_rad_x * rng
        dy = flow_rad_y * rng
        step = math.hypot(dx, dy)
        pos_x += dx
        pos_y += dy
        path_length += step

    if t0 is None or last_t is None:
        return DriftStats(0.0, 0.0, 0.0, 0.0, 0.0)

    duration = max(0.0, last_t - t0)
    total_distance = math.hypot(pos_x, pos_y)
    drift_per_min = total_distance / (duration / 60.0) if duration > 0.0 else 0.0
    drift_per_meter = total_distance / path_length if path_length > 0.0 else 0.0
    return DriftStats(duration, total_distance, path_length, drift_per_min, drift_per_meter)


def _collect_paths(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root]
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".csv"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="CSV file or directory containing flight logs")
    args = ap.parse_args()

    paths = _collect_paths(args.path)
    if not paths:
        print("No CSV logs found.")
        return 1

    stats = compute_drift(paths)
    print(f"duration_s: {stats.duration_s:.2f}")
    print(f"total_distance_m: {stats.total_distance_m:.3f}")
    print(f"path_length_m: {stats.path_length_m:.3f}")
    print(f"drift_per_minute_m: {stats.drift_per_minute_m:.3f}")
    print(f"drift_per_meter: {stats.drift_per_meter:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
