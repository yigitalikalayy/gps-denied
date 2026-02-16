from __future__ import annotations

import argparse
import collections
import math
import statistics
import threading
import time

from .camera import Camera
from .config import AppConfig
from .flow_estimator import OpticalFlowEstimator
from .stats import Rate


def _monotonic_to_time_usec(t0_monotonic: float, t0_time: float, t_monotonic: float) -> int:
    return int((t0_time + (t_monotonic - t0_monotonic)) * 1_000_000)


def _monotonic_to_time_boot_ms(t0_monotonic: float, t_monotonic: float) -> int:
    return int((t_monotonic - t0_monotonic) * 1_000.0)


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
    sync_cfg = cfg.section("time_sync")

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

            def send_distance_sensor(self, **_kwargs):
                return None

            def read_yaw_samples(self, max_samples: int = 0, max_age_s: float = 0.0):
                _ = max_samples
                _ = max_age_s
                return []

            def close(self):
                return None

        bridge = _NullBridge()

    sync = None
    sync_enabled = bool(sync_cfg.get("enabled", False))
    sync_cls = None
    sync_min_geom_inliers = max(0, int(sync_cfg.get("min_geom_inliers", 8)))
    sync_max_abs_yaw_delta = max(0.0, float(sync_cfg.get("max_abs_yaw_delta_rad", 0.05)))
    if sync_enabled and serial_enabled:
        from .time_sync import MotionBasedTimeSync

        sync_cls = MotionBasedTimeSync

    lidar = None
    if bool(lidar_cfg.get("enabled", True)):
        backend = str(lidar_cfg.get("backend", "lw20_ascii"))
        if backend == "lw20_ascii":
            from .lidar import LightwareLw20AsciiI2C

            lidar = LightwareLw20AsciiI2C(lidar_cfg)
        elif backend in ("lw20_binary", "lw20b_binary"):
            from .lidar import LightwareLw20BinaryI2C

            lidar = LightwareLw20BinaryI2C(lidar_cfg)
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
    lidar_median_window = int(lidar_filter_cfg.get("median_window", 5))
    if lidar_median_window < 1:
        lidar_median_window = 1
    if (lidar_median_window % 2) == 0:
        lidar_median_window += 1
    lidar_max_jump_m = float(lidar_filter_cfg.get("max_jump_m", 0.0))
    if lidar_max_jump_m < 0.0:
        lidar_max_jump_m = 0.0
    lidar_reset_s = float(lidar_filter_cfg.get("reset_s", 0.2))
    if lidar_reset_s < 0.0:
        lidar_reset_s = 0.0
    lidar_reject_limit = int(lidar_filter_cfg.get("reject_limit", 3))
    if lidar_reject_limit < 0:
        lidar_reject_limit = 0
    lidar_max_dt_s = float(lidar_filter_cfg.get("max_dt_s", 0.0))
    if lidar_max_dt_s < 0.0:
        lidar_max_dt_s = 0.0
    lidar_debug = bool(lidar_cfg.get("debug", False))
    lidar_mav_cfg = lidar_cfg.get("mavlink", {}) if isinstance(lidar_cfg.get("mavlink", {}), dict) else {}
    dist_sensor_type = int(lidar_mav_cfg.get("type", 0))
    dist_sensor_id = int(lidar_mav_cfg.get("id", 0))
    dist_sensor_orientation = int(lidar_mav_cfg.get("orientation", 25))
    dist_sensor_covariance = int(lidar_mav_cfg.get("covariance", 255))
    dist_sensor_min_m = float(lidar_mav_cfg.get("min_distance_m", lidar_min_m))
    dist_sensor_max_m = float(lidar_mav_cfg.get("max_distance_m", lidar_max_m))

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

    sync_nominal_source = str(sync_cfg.get("nominal_fps_source", "runtime")).strip().lower()
    sync_bootstrap_frames = max(5, int(sync_cfg.get("bootstrap_frames", 60)))
    sync_bootstrap_timeout_s = max(0.2, float(sync_cfg.get("bootstrap_timeout_s", 2.0)))
    sync_bootstrap_start_t = prev.t_monotonic
    sync_fps_samples: collections.deque[float] = collections.deque(maxlen=sync_bootstrap_frames * 2)
    sync_nominal_cfg_fps = max(1.0, float(cam_cfg.get("fps", 60.0)))
    if sync_cls is not None and sync_nominal_source == "config":
        sync = sync_cls(sync_cfg, imu_source=bridge, nominal_fps_hz=sync_nominal_cfg_fps)

    cam_k = cam.camera_matrix()
    if cam_k is None:
        import numpy as np  # type: ignore

        h0, w0 = prev.gray.shape[:2]
        cx = 0.5 * float(max(0, w0 - 1))
        cy = 0.5 * float(max(0, h0 - 1))
        cam_k = np.array(
            [
                [float(focal_length_px), 0.0, cx],
                [0.0, float(focal_length_px), cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    flow.set_camera_intrinsics(cam_k)

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
    last_lidar_used_prev = None
    last_lidar_used_prev_t = None
    lidar_ema = None
    lidar_raw_window = collections.deque(maxlen=lidar_median_window)
    last_lidar_error_t = None
    lidar_lock = threading.Lock()
    lidar_thread_stop = threading.Event()
    lidar_thread = None
    lidar_reject_count = 0

    def _lidar_poll_loop() -> None:
        nonlocal last_lidar, last_lidar_t, last_lidar_valid, last_lidar_valid_t
        nonlocal last_lidar_used, last_lidar_used_t, lidar_ema, last_lidar_error_t

        if lidar_poll_hz <= 0.0:
            return
        poll_period = 1.0 / max(1.0, lidar_poll_hz)

        nonlocal lidar_reject_count

        while not lidar_thread_stop.is_set():
            loop_start = time.monotonic()

            if lidar is not None:
                try:
                    sample = lidar.read()
                    d_raw = float(sample.distance_m)
                    t_raw = float(sample.t_monotonic)

                    with lidar_lock:
                        last_lidar = sample
                        last_lidar_t = t_raw

                        valid = True
                        d_use = d_raw
                        if d_raw < lidar_min_m or d_raw > lidar_max_m:
                            valid = False
                            if lidar_debug and (
                                last_lidar_error_t is None or (loop_start - last_lidar_error_t) > 1.0
                            ):
                                print(f"lidar rejected: out_of_range d_raw={d_raw:.3f}m")
                                last_lidar_error_t = loop_start

                        if valid and lidar_median_window > 1:
                            values = list(lidar_raw_window)
                            values.append(d_raw)
                            if len(values) > lidar_median_window:
                                values.pop(0)
                            values.sort()
                            d_use = values[len(values) // 2]

                        if (
                            valid
                            and lidar_max_jump_m > 0.0
                            and last_lidar_valid is not None
                            and abs(d_use - last_lidar_valid) > lidar_max_jump_m
                        ):
                            valid = False
                            if lidar_debug and (
                                last_lidar_error_t is None or (loop_start - last_lidar_error_t) > 1.0
                            ):
                                print(
                                    "lidar rejected: jump"
                                    f" d_use={d_use:.3f}m last={last_lidar_valid:.3f}m"
                                )
                                last_lidar_error_t = loop_start

                        if (
                            valid
                            and last_lidar_valid is not None
                            and last_lidar_valid_t is not None
                            and lidar_max_speed > 0.0
                        ):
                            dt_lidar = max(1e-3, t_raw - last_lidar_valid_t)
                            if lidar_max_dt_s > 0.0:
                                dt_lidar = min(dt_lidar, lidar_max_dt_s)
                            if abs(d_use - last_lidar_valid) > (lidar_max_speed * dt_lidar):
                                valid = False
                                if lidar_debug and (
                                    last_lidar_error_t is None or (loop_start - last_lidar_error_t) > 1.0
                                ):
                                    print(
                                        "lidar rejected: speed"
                                        f" d_use={d_use:.3f}m last={last_lidar_valid:.3f}m dt={dt_lidar:.3f}s"
                                    )
                                    last_lidar_error_t = loop_start

                        if not valid:
                            lidar_reject_count += 1
                            allow_reset = False
                            if lidar_reset_s > 0.0 and last_lidar_valid_t is not None:
                                if (t_raw - last_lidar_valid_t) >= lidar_reset_s:
                                    allow_reset = True
                            if lidar_reject_limit > 0 and lidar_reject_count >= lidar_reject_limit:
                                allow_reset = True
                            if allow_reset:
                                valid = True
                                lidar_reject_count = 0
                                lidar_raw_window.clear()
                                if lidar_debug and (
                                    last_lidar_error_t is None or (loop_start - last_lidar_error_t) > 1.0
                                ):
                                    print("lidar reset: accepting new value after rejects")
                                    last_lidar_error_t = loop_start

                        if valid:
                            lidar_reject_count = 0
                            lidar_raw_window.append(d_raw)
                            last_lidar_valid = d_use
                            last_lidar_valid_t = t_raw
                            if lidar_ema_alpha > 0.0:
                                if lidar_ema is None:
                                    lidar_ema = d_use
                                else:
                                    lidar_ema += lidar_ema_alpha * (d_use - lidar_ema)
                                if last_lidar_used is not None:
                                    last_lidar_used_prev = last_lidar_used
                                    last_lidar_used_prev_t = last_lidar_used_t
                                last_lidar_used = float(lidar_ema)
                            else:
                                if last_lidar_used is not None:
                                    last_lidar_used_prev = last_lidar_used
                                    last_lidar_used_prev_t = last_lidar_used_t
                                last_lidar_used = d_use
                            last_lidar_used_t = t_raw
                except Exception as exc:
                    if lidar_debug:
                        if last_lidar_error_t is None or (loop_start - last_lidar_error_t) > 1.0:
                            print(f"lidar read failed: {exc}")
                            last_lidar_error_t = loop_start

            elapsed = time.monotonic() - loop_start
            sleep_s = poll_period - elapsed
            if sleep_s > 0.0:
                lidar_thread_stop.wait(sleep_s)

    if lidar is not None and lidar_poll_hz > 0.0:
        lidar_thread = threading.Thread(target=_lidar_poll_loop, name="lidar-poll", daemon=True)
        lidar_thread.start()

    try:
        while True:
            fr = fq.get_latest() if fq is not None else cam.read()
            frame_count += 1

            fps = frame_rate.tick(fr.t_monotonic)
            dt_s = max(1e-6, fr.t_monotonic - prev.t_monotonic)
            dt_us = int(dt_s * 1_000_000)

            if sync_cls is not None and sync is None:
                if math.isfinite(fps) and fps > 1.0:
                    sync_fps_samples.append(float(fps))
                fps_ready = len(sync_fps_samples) >= sync_bootstrap_frames
                fps_timed_out = (fr.t_monotonic - sync_bootstrap_start_t) >= sync_bootstrap_timeout_s
                if fps_ready or fps_timed_out:
                    if sync_fps_samples:
                        nominal_fps = float(statistics.median(sync_fps_samples))
                    else:
                        nominal_fps = sync_nominal_cfg_fps
                    nominal_fps = max(1.0, nominal_fps)
                    sync = sync_cls(sync_cfg, imu_source=bridge, nominal_fps_hz=nominal_fps)
                    print(
                        "time_sync initialized "
                        f"nominal_fps={nominal_fps:.2f}Hz source={'runtime' if sync_fps_samples else 'config_fallback'}"
                    )

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
            if sync is not None:
                yaw_delta_for_sync = None
                if r.yaw_rad is not None:
                    yaw_candidate = float(r.yaw_rad)
                    if (
                        math.isfinite(yaw_candidate)
                        and r.geom_inliers >= sync_min_geom_inliers
                        and abs(yaw_candidate) <= sync_max_abs_yaw_delta
                    ):
                        yaw_delta_for_sync = yaw_candidate
                if yaw_delta_for_sync is not None:
                    sync.push_camera_yaw_delta(
                        frame_id=frame_count,
                        yaw_delta_rad=yaw_delta_for_sync,
                        t_monotonic=fr.t_monotonic,
                        motion_hint=max(abs(g.z_rad_s), abs(r.motion_mag)),
                    )

            distance_m = 0.0
            time_delta_distance_us = 0
            distance_valid = False
            if lidar is not None:
                with lidar_lock:
                    used = last_lidar_used
                    used_t = last_lidar_used_t
                    used_prev = last_lidar_used_prev
                    used_prev_t = last_lidar_used_prev_t
                if used_t is not None and used_t > fr.t_monotonic:
                    if used_prev_t is not None and used_prev is not None:
                        used = used_prev
                        used_t = used_prev_t
                if used is not None and used_t is not None:
                    distance_m = float(used)
                    time_delta_distance_us = int(max(0.0, fr.t_monotonic - used_t) * 1_000_000)
                    distance_valid = True

            now = fr.t_monotonic
            if (now - last_pub_t) >= pub_period and integration_us > 0:
                quality = int(round(quality_acc / max(1, quality_n)))
                time_usec = _monotonic_to_time_usec(t0_mono, t0_time, now)
                if sync is not None:
                    sync_usec = sync.estimate_frame_time_usec(frame_count)
                    if sync_usec is not None:
                        time_usec = int(sync_usec)
                time_boot_ms = _monotonic_to_time_boot_ms(t0_mono, now)

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

                if distance_valid:
                    bridge.send_distance_sensor(
                        time_boot_ms=time_boot_ms,
                        min_distance_m=float(dist_sensor_min_m),
                        max_distance_m=float(dist_sensor_max_m),
                        current_distance_m=float(distance_m),
                        sensor_type=dist_sensor_type,
                        sensor_id=dist_sensor_id,
                        orientation=dist_sensor_orientation,
                        covariance=dist_sensor_covariance,
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
                sync_info = ""
                if sync is not None:
                    s = sync.read_state()
                    sync_info = (
                        f" sync(status={s.health_status} score={s.health_score:.2f} reason={s.health_reason}"
                        f" off={s.t_offset_s:+.6f}s fps={s.fps_hz:7.3f}"
                        f" lag_ms={1000.0 * s.lag_s:+6.2f} corr={s.corr_peak:.3f}"
                        f" good={s.good_updates} bad={s.bad_updates})"
                    )
                if print_raw_flow:
                    if quality_ema_alpha > 0.0:
                        print(
                            f"frames={frame_count} fps={fps:6.1f} "
                            f"q={q_use:3d} q_raw={q_raw:3d} tracked={r.tracked:3d} "
                            f"geom_in={r.geom_inliers:3d} yaw_d={0.0 if r.yaw_rad is None else r.yaw_rad:+.4f} "
                            f"dx_px={r.dx_px:+7.3f} dy_px={r.dy_px:+7.3f} "
                            f"int(rad)=({integrated_x:+.5f},{integrated_y:+.5f}) t_int_us={integration_us:7d} "
                            f"int_g(rad)=({integrated_xgyro:+.5f},{integrated_ygyro:+.5f},{integrated_zgyro:+.5f}) "
                            f"g=({g.x_rad_s:+6.3f},{g.y_rad_s:+6.3f},{g.z_rad_s:+6.3f}) "
                            f"lidar={distance_m:5.2f}m dt_lidar_us={time_delta_distance_us:7d}"
                            f"{sync_info}"
                        )
                    else:
                        print(
                            f"frames={frame_count} fps={fps:6.1f} "
                            f"q={q_raw:3d} tracked={r.tracked:3d} "
                            f"geom_in={r.geom_inliers:3d} yaw_d={0.0 if r.yaw_rad is None else r.yaw_rad:+.4f} "
                            f"dx_px={r.dx_px:+7.3f} dy_px={r.dy_px:+7.3f} "
                            f"int(rad)=({integrated_x:+.5f},{integrated_y:+.5f}) t_int_us={integration_us:7d} "
                            f"int_g(rad)=({integrated_xgyro:+.5f},{integrated_ygyro:+.5f},{integrated_zgyro:+.5f}) "
                            f"g=({g.x_rad_s:+6.3f},{g.y_rad_s:+6.3f},{g.z_rad_s:+6.3f}) "
                            f"lidar={distance_m:5.2f}m dt_lidar_us={time_delta_distance_us:7d}"
                            f"{sync_info}"
                        )
                else:
                    print(f"frames={frame_count} fps={fps:6.1f} q={q_use:3d} lidar={distance_m:5.2f}m{sync_info}")

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
            lidar_thread_stop.set()
            if lidar_thread is not None:
                lidar_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        try:
            if sync is not None:
                sync.close()
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
