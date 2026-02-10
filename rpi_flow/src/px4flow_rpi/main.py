from __future__ import annotations

import argparse
import collections
import threading
import time

from .camera import Camera
from .config import AppConfig
from .flow_estimator import OpticalFlowEstimator
from .stats import Rate


def _monotonic_to_time_usec(t0_monotonic: float, t0_time: float, t_monotonic: float) -> int:
    return int((t0_time + (t_monotonic - t0_monotonic)) * 1_000_000)


class _FrameQueue:
    def __init__(self, cam: Camera, maxlen: int) -> None:
        self._cam = cam
        self._q: collections.deque = collections.deque(maxlen=max(1, int(maxlen)))
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="camera-capture", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop:
            try:
                fr = self._cam.read()
            except Exception:
                time.sleep(0.01)
                continue
            with self._cv:
                self._q.append(fr)
                self._cv.notify_all()

    def get_latest(self, timeout_s: float = 1.0):
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._cv:
            while not self._q and not self._stop:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
            if not self._q:
                raise RuntimeError("camera timeout")
            fr = self._q.pop()
            self._q.clear()
            return fr

    def close(self) -> None:
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config.json")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    serial_cfg = cfg.section("serial")
    cam_cfg = cfg.section("camera")
    flow_cfg = cfg.section("flow")
    gyro_cfg = cfg.section("gyro")
    lidar_cfg = cfg.section("lidar")
    log_cfg = cfg.section("logging")

    publish_hz = float(flow_cfg.get("publish_hz", 50))
    focal_length_px = float(flow_cfg.get("focal_length_px", 800.0))
    axis_swap_xy = bool(flow_cfg.get("axis_swap_xy", True))
    axis_sign_x = float(flow_cfg.get("axis_sign_x", 1.0))
    axis_sign_y = float(flow_cfg.get("axis_sign_y", -1.0))
    quality_ema_alpha = float(flow_cfg.get("quality_ema_alpha", 0.0))
    quality_ema_alpha = max(0.0, min(1.0, quality_ema_alpha))

    gyro_sign_x = float(gyro_cfg.get("axis_sign_x", 1.0))
    gyro_sign_y = float(gyro_cfg.get("axis_sign_y", 1.0))
    gyro_sign_z = float(gyro_cfg.get("axis_sign_z", 1.0))

    print_hz = float(log_cfg.get("print_hz", 10))
    print_raw_flow = bool(log_cfg.get("print_raw_flow", True))

    # MAVLink message uses sensor_id (u8). PX4Flow uses a param; here we default to sysid.
    sensor_id = int(serial_cfg.get("sysid", 42)) & 0xFF

    cam = Camera(cam_cfg)
    flow = OpticalFlowEstimator(flow_cfg)

    serial_enabled = bool(serial_cfg.get("enabled", True))
    if serial_enabled:
        from .mavlink_bridge import MavlinkBridge  # local import: allow PC testing without pyserial

        bridge = MavlinkBridge(serial_cfg)
    else:
        class _NullBridge:
            def read_gyro(self):
                class _G:
                    x_rad_s = 0.0
                    y_rad_s = 0.0
                    z_rad_s = 0.0
                    temperature_c = 0.0

                return _G()

            def send_optical_flow_rad(self, **_kwargs):
                return None

            def close(self):
                return None

        bridge = _NullBridge()

    lidar = None
    if bool(lidar_cfg.get("enabled", True)):
        backend = str(lidar_cfg.get("backend", "lw20_ascii"))
        if backend == "lw20_ascii":
            from .lidar import LightwareLw20AsciiI2C

            lidar = LightwareLw20AsciiI2C(lidar_cfg)
        else:
            raise ValueError(f"unsupported lidar backend: {backend}")
        lidar_poll_hz = float(lidar_cfg.get("poll_hz", 50))
    else:
        lidar_poll_hz = 0.0
    lidar_filter_cfg = lidar_cfg.get("filter", {}) if isinstance(lidar_cfg.get("filter", {}), dict) else {}
    lidar_min_m = float(lidar_filter_cfg.get("min_m", 0.2))
    lidar_max_m = float(lidar_filter_cfg.get("max_m", 50.0))
    lidar_max_speed = float(lidar_filter_cfg.get("max_speed_m_s", 5.0))
    lidar_ema_alpha = float(lidar_filter_cfg.get("ema_alpha", 0.0))
    lidar_ema_alpha = max(0.0, min(1.0, lidar_ema_alpha))

    t0_mono = time.monotonic()
    t0_time = time.time()

    frame_rate = Rate(alpha=0.05)
    pub_period = 1.0 / max(1e-3, publish_hz)
    log_period = 1.0 / max(1e-3, print_hz)

    threaded = bool(cam_cfg.get("threaded", True))
    queue_size = int(cam_cfg.get("queue_size", 2))
    fq = _FrameQueue(cam, maxlen=queue_size) if threaded else None

    prev = fq.get_latest() if fq is not None else cam.read()
    frame_count = 0

    last_pub_t = prev.t_monotonic
    last_log_t = prev.t_monotonic

    integration_us = 0
    integrated_x = 0.0
    integrated_y = 0.0
    integrated_xgyro = 0.0
    integrated_ygyro = 0.0
    integrated_zgyro = 0.0
    quality_acc = 0
    quality_n = 0
    quality_ema = None

    last_lidar = None
    last_lidar_t = None
    last_lidar_valid = None
    last_lidar_valid_t = None
    last_lidar_used = None
    last_lidar_used_t = None
    lidar_ema = None

    try:
        while True:
            fr = fq.get_latest() if fq is not None else cam.read()
            frame_count += 1

            fps = frame_rate.tick(fr.t_monotonic)
            dt_s = max(1e-6, fr.t_monotonic - prev.t_monotonic)
            dt_us = int(dt_s * 1_000_000)

            r = flow.estimate(prev.gray, fr.gray)

            # Match PX4Flow firmware mapping (axis swap + sign) with configurable parameters.
            if axis_swap_xy:
                pix_x = r.dy_px
                pix_y = r.dx_px
            else:
                pix_x = r.dx_px
                pix_y = r.dy_px

            integrated_x += axis_sign_x * (pix_x / focal_length_px)
            integrated_y += axis_sign_y * (pix_y / focal_length_px)
            integration_us += dt_us
            q_raw = int(r.quality_u8)
            if quality_ema_alpha > 0.0:
                if quality_ema is None:
                    quality_ema = float(q_raw)
                else:
                    quality_ema += quality_ema_alpha * (q_raw - quality_ema)
                q_use = int(round(quality_ema))
            else:
                q_use = q_raw

            quality_acc += int(q_use)
            quality_n += 1

            g = bridge.read_gyro()
            integrated_xgyro += gyro_sign_x * g.x_rad_s * dt_s
            integrated_ygyro += gyro_sign_y * g.y_rad_s * dt_s
            integrated_zgyro += gyro_sign_z * g.z_rad_s * dt_s

            distance_m = 0.0
            time_delta_distance_us = 0
            if lidar is not None:
                if last_lidar_t is None or (fr.t_monotonic - last_lidar_t) >= (1.0 / max(1.0, lidar_poll_hz)):
                    try:
                        last_lidar = lidar.read()
                        last_lidar_t = last_lidar.t_monotonic
                    except Exception:
                        pass
                if last_lidar is not None and last_lidar_t is not None:
                    d_raw = float(last_lidar.distance_m)
                    t_raw = float(last_lidar.t_monotonic)
                    valid = True
                    if d_raw < lidar_min_m or d_raw > lidar_max_m:
                        valid = False
                    if (
                        valid
                        and last_lidar_valid is not None
                        and last_lidar_valid_t is not None
                        and lidar_max_speed > 0.0
                    ):
                        dt_lidar = max(1e-3, t_raw - last_lidar_valid_t)
                        if abs(d_raw - last_lidar_valid) > (lidar_max_speed * dt_lidar):
                            valid = False
                    if valid:
                        last_lidar_valid = d_raw
                        last_lidar_valid_t = t_raw
                        if lidar_ema_alpha > 0.0:
                            if lidar_ema is None:
                                lidar_ema = d_raw
                            else:
                                lidar_ema += lidar_ema_alpha * (d_raw - lidar_ema)
                            last_lidar_used = float(lidar_ema)
                        else:
                            last_lidar_used = d_raw
                        last_lidar_used_t = t_raw
                if last_lidar_used is not None and last_lidar_used_t is not None:
                    distance_m = float(last_lidar_used)
                    time_delta_distance_us = int((fr.t_monotonic - last_lidar_used_t) * 1_000_000)

            now = fr.t_monotonic
            if (now - last_pub_t) >= pub_period and integration_us > 0:
                quality = int(round(quality_acc / max(1, quality_n)))
                time_usec = _monotonic_to_time_usec(t0_mono, t0_time, now)

                bridge.send_optical_flow_rad(
                    time_usec=time_usec,
                    sensor_id=sensor_id,
                    integration_time_us=int(integration_us),
                    integrated_x=float(integrated_x),
                    integrated_y=float(integrated_y),
                    integrated_xgyro=float(integrated_xgyro),
                    integrated_ygyro=float(integrated_ygyro),
                    integrated_zgyro=float(integrated_zgyro),
                    temperature=float(g.temperature_c),
                    quality=int(quality),
                    time_delta_distance_us=int(time_delta_distance_us),
                    distance_m=float(distance_m),
                )

                integration_us = 0
                integrated_x = 0.0
                integrated_y = 0.0
                integrated_xgyro = 0.0
                integrated_ygyro = 0.0
                integrated_zgyro = 0.0
                quality_acc = 0
                quality_n = 0
                last_pub_t = now

            if (now - last_log_t) >= log_period:
                last_log_t = now
                if print_raw_flow:
                    if quality_ema_alpha > 0.0:
                        print(
                            f"frames={frame_count} fps={fps:6.1f} "
                            f"q={q_use:3d} q_raw={q_raw:3d} tracked={r.tracked:3d} "
                            f"dx_px={r.dx_px:+7.3f} dy_px={r.dy_px:+7.3f} "
                            f"int(rad)=({integrated_x:+.5f},{integrated_y:+.5f}) t_int_us={integration_us:7d} "
                            f"int_g(rad)=({integrated_xgyro:+.5f},{integrated_ygyro:+.5f},{integrated_zgyro:+.5f}) "
                            f"g=({g.x_rad_s:+6.3f},{g.y_rad_s:+6.3f},{g.z_rad_s:+6.3f}) "
                            f"lidar={distance_m:5.2f}m dt_lidar_us={time_delta_distance_us:7d}"
                        )
                    else:
                        print(
                            f"frames={frame_count} fps={fps:6.1f} "
                            f"q={q_raw:3d} tracked={r.tracked:3d} "
                            f"dx_px={r.dx_px:+7.3f} dy_px={r.dy_px:+7.3f} "
                            f"int(rad)=({integrated_x:+.5f},{integrated_y:+.5f}) t_int_us={integration_us:7d} "
                            f"int_g(rad)=({integrated_xgyro:+.5f},{integrated_ygyro:+.5f},{integrated_zgyro:+.5f}) "
                            f"g=({g.x_rad_s:+6.3f},{g.y_rad_s:+6.3f},{g.z_rad_s:+6.3f}) "
                            f"lidar={distance_m:5.2f}m dt_lidar_us={time_delta_distance_us:7d}"
                        )
                else:
                    print(f"frames={frame_count} fps={fps:6.1f} q={q_use:3d} lidar={distance_m:5.2f}m")

            prev = fr
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            if fq is not None:
                fq.close()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        try:
            bridge.close()
        except Exception:
            pass
        try:
            if lidar is not None:
                lidar.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
