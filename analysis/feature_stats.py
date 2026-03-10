from __future__ import annotations

import argparse
import csv
import os
from statistics import mean, pstdev
from typing import List


def _collect_paths(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root]
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".csv"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def _as_float(row, key, default=None):
    try:
        v = row.get(key)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _linear_trend(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="CSV file or directory containing flight logs")
    args = ap.parse_args()

    paths = _collect_paths(args.path)
    if not paths:
        print("No CSV logs found.")
        return 1

    features = []
    tracked = []
    loss_rates = []
    times = []
    flow_x = []
    flow_y = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = _as_float(row, "timestamp_s")
                fc = _as_float(row, "feature_count")
                tr = _as_float(row, "tracked_feature_count")
                lr = _as_float(row, "tracking_loss_rate")
                fx = _as_float(row, "flow_rad_x")
                fy = _as_float(row, "flow_rad_y")
                if t is None or fc is None or tr is None or lr is None:
                    continue
                times.append(t)
                features.append(fc)
                tracked.append(tr)
                loss_rates.append(lr)
                if fx is not None:
                    flow_x.append(fx)
                if fy is not None:
                    flow_y.append(fy)

    if not features:
        print("No feature data found.")
        return 1

    feature_decay = _linear_trend(times, tracked)

    print(f"feature_count: mean={mean(features):.2f} std={pstdev(features):.2f}")
    print(f"tracked_feature_count: mean={mean(tracked):.2f} std={pstdev(tracked):.2f}")
    print(f"tracking_loss_rate: mean={mean(loss_rates):.3f} std={pstdev(loss_rates):.3f}")
    print(f"feature_decay (tracked/s): {feature_decay:.3f}")
    if flow_x and flow_y:
        flow_noise = (pstdev(flow_x) ** 2 + pstdev(flow_y) ** 2) ** 0.5
        print(f"flow_noise (rad): {flow_noise:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
