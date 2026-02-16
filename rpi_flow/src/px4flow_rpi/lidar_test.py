from __future__ import annotations

import argparse
import time
from typing import Any

from .config import AppConfig
from .lidar import LightwareLw20AsciiI2C, LightwareLw20BinaryI2C


def _build_lidar(cfg: dict[str, Any]):
    backend = str(cfg.get("backend", "lw20_ascii"))
    if backend == "lw20_ascii":
        return LightwareLw20AsciiI2C(cfg)
    if backend in ("lw20_binary", "lw20b_binary"):
        return LightwareLw20BinaryI2C(cfg)
    raise ValueError(f"unsupported lidar backend: {backend}")


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.json")
    ap.add_argument("--hz", type=float, default=0.0, help="override lidar.poll_hz (0 = no pacing)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to run (0 = forever)")
    ap.add_argument("--print-every", type=int, default=1, help="print every N samples")
    ap.add_argument("--raw", action="store_true", help="print raw response for ascii backend")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    lidar_cfg = cfg.section("lidar")

    poll_hz = float(lidar_cfg.get("poll_hz", 50))
    if args.hz > 0.0:
        poll_hz = args.hz

    lidar = _build_lidar(lidar_cfg)

    print(
        "[lidar_test] backend="
        f"{lidar_cfg.get('backend', 'lw20_ascii')} bus={lidar_cfg.get('i2c_bus', 1)} "
        f"addr=0x{int(lidar_cfg.get('i2c_address', 0x66)):02x} poll_hz={poll_hz}"
    )

    period = 0.0 if poll_hz <= 0.0 else 1.0 / max(1.0, poll_hz)
    start = time.monotonic()
    last_t = None
    samples = 0
    errors = 0
    dt_hist: list[float] = []
    read_hist: list[float] = []

    try:
        while True:
            loop_start = time.monotonic()
            try:
                sample = lidar.read()
                samples += 1
                t = float(sample.t_monotonic)
                dt = None if last_t is None else (t - last_t)
                last_t = t
                read_ms = (time.monotonic() - loop_start) * 1000.0
                read_hist.append(read_ms)

                if dt is not None:
                    dt_hist.append(dt)
                if samples % max(1, int(args.print_every)) == 0:
                    hz = 0.0 if dt is None or dt <= 0.0 else (1.0 / dt)
                    line = (
                        f"n={samples:6d} d={sample.distance_m:7.3f}m "
                        f"dt={0.0 if dt is None else dt*1000.0:7.2f}ms "
                        f"hz={hz:6.1f} read={read_ms:6.2f}ms"
                    )
                    if args.raw and hasattr(lidar, "last_raw"):
                        line += f" raw={getattr(lidar, 'last_raw', '')!r}"
                    print(line)
            except Exception as exc:
                errors += 1
                print(f"[lidar_test] read failed: {exc}")

            if period > 0.0:
                elapsed = time.monotonic() - loop_start
                sleep_s = period - elapsed
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

            if args.duration > 0.0 and (time.monotonic() - start) >= args.duration:
                break
    finally:
        try:
            lidar.close()
        except Exception:
            pass

    if dt_hist:
        hz_avg = 1.0 / max(1e-6, _avg(dt_hist))
        dt_ms = [v * 1000.0 for v in dt_hist]
        print(
            f"[lidar_test] samples={samples} errors={errors} "
            f"avg_hz={hz_avg:.1f} dt_ms(min/avg/max)={min(dt_ms):.2f}/"
            f"{_avg(dt_ms):.2f}/{max(dt_ms):.2f} "
            f"read_ms(avg)={_avg(read_hist):.2f}"
        )
    else:
        print(f"[lidar_test] samples={samples} errors={errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
