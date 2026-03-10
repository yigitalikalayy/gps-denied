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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="CSV file or directory containing flight logs")
    args = ap.parse_args()

    paths = _collect_paths(args.path)
    if not paths:
        print("No CSV logs found.")
        return 1

    ranges = []
    delays = []
    invalid = 0
    total = 0

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                valid = str(row.get("range_valid", "")).lower() in ("true", "1", "yes")
                if not valid:
                    invalid += 1
                    continue
                rng = _as_float(row, "filtered_range")
                if rng is not None:
                    ranges.append(rng)
                delay = _as_float(row, "range_delay_us")
                if delay is not None:
                    delays.append(delay / 1e3)

    if not ranges:
        print("No valid range data found.")
        return 1

    print(f"samples: {len(ranges)} (invalid: {invalid} / total: {total})")
    print(f"range_m: min={min(ranges):.3f} max={max(ranges):.3f} mean={mean(ranges):.3f} std={pstdev(ranges):.3f}")
    if delays:
        print(f"range_delay_ms: mean={mean(delays):.2f} std={pstdev(delays):.2f} min={min(delays):.2f} max={max(delays):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
