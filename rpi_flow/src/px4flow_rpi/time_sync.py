from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _CameraYawSample:
    frame_id: int
    yaw_cum_rad: float
    t_monotonic: float
    motion_hint: float


@dataclass
class TimeSyncState:
    ready: bool
    t_offset_s: float
    fps_hz: float
    lag_s: float
    corr_peak: float
    last_update_monotonic: float
    health_status: str
    health_score: float
    health_reason: str
    good_updates: int
    bad_updates: int


class MotionBasedTimeSync:
    def __init__(self, cfg: dict[str, Any], imu_source: Any, nominal_fps_hz: float) -> None:
        import numpy as np  # type: ignore

        self._enabled = bool(cfg.get("enabled", False))
        self._imu_source = imu_source
        self._nominal_fps_hz = max(1.0, float(nominal_fps_hz))
        self._update_hz = max(1.0, float(cfg.get("update_hz", 20.0)))
        self._cam_window = max(16, int(cfg.get("camera_window_samples", 240)))
        self._imu_window = max(64, int(cfg.get("imu_window_samples", 2400)))
        self._imu_max_age_s = max(0.5, float(cfg.get("imu_max_age_s", 10.0)))
        self._min_samples = max(16, int(cfg.get("min_samples", 48)))
        self._min_angular_rate = max(0.0, float(cfg.get("min_angular_rate_rad_s", 0.08)))
        self._min_signal_std = max(1e-6, float(cfg.get("min_signal_std", 1e-4)))
        self._max_lag_s = max(0.0, float(cfg.get("max_lag_s", 0.3)))
        self._min_corr_peak = float(cfg.get("min_corr_peak", 0.05))
        self._sigma_s = max(1e-6, float(cfg.get("measurement_sigma_s", 0.003)))

        self._num_particles = max(16, int(cfg.get("particle_count", 256)))
        self._init_offset_std_s = max(1e-5, float(cfg.get("init_offset_std_s", 0.01)))
        self._init_fps_std_hz = max(1e-4, float(cfg.get("init_fps_std_hz", 1.0)))
        self._proc_offset_noise_s = max(0.0, float(cfg.get("process_noise_offset_s", 0.0005)))
        self._proc_fps_noise_hz = max(0.0, float(cfg.get("process_noise_fps_hz", 0.02)))
        self._fps_min = max(1.0, float(cfg.get("fps_min_hz", self._nominal_fps_hz * 0.5)))
        self._fps_max = max(self._fps_min + 1e-3, float(cfg.get("fps_max_hz", self._nominal_fps_hz * 1.5)))

        health_cfg = cfg.get("health", {}) if isinstance(cfg.get("health", {}), dict) else {}
        self._health_min_corr = max(1e-6, float(health_cfg.get("min_corr_peak", max(self._min_corr_peak, 0.1))))
        self._health_max_abs_lag_s = max(1e-6, float(health_cfg.get("max_abs_lag_s", 0.008)))
        self._health_max_fps_error_rel = max(1e-6, float(health_cfg.get("max_fps_error_rel", 0.03)))
        self._health_min_good_updates = max(1, int(health_cfg.get("min_good_updates", 5)))
        self._health_max_stale_s = max(0.1, float(health_cfg.get("max_stale_s", 1.5)))
        self._health_ema_alpha = min(1.0, max(0.0, float(health_cfg.get("ema_alpha", 0.2))))
        self._corr_ema = 0.0
        self._lag_abs_ema_s = 0.0
        self._fps_err_ema = 0.0
        self._good_updates = 0
        self._bad_updates = 0
        self._health_has_sample = False

        self._cam_lock = threading.Lock()
        self._cam_samples: collections.deque[_CameraYawSample] = collections.deque(maxlen=self._cam_window)
        self._cam_yaw_cum = 0.0
        self._last_frame_id = -1

        self._state_lock = threading.Lock()
        self._state = TimeSyncState(
            ready=False,
            t_offset_s=0.0,
            fps_hz=self._nominal_fps_hz,
            lag_s=0.0,
            corr_peak=0.0,
            last_update_monotonic=0.0,
            health_status="warming_up",
            health_score=0.0,
            health_reason="insufficient_data",
            good_updates=0,
            bad_updates=0,
        )

        self._particles = None
        self._weights = None
        self._rng = np.random.default_rng(int(time.time() * 1e9) & 0xFFFFFFFF)
        self._stop = threading.Event()
        self._thread = None

        if self._enabled:
            self._thread = threading.Thread(target=self._loop, name="time-sync", daemon=True)
            self._thread.start()

    def push_camera_yaw_delta(
        self,
        *,
        frame_id: int,
        yaw_delta_rad: float | None,
        t_monotonic: float,
        motion_hint: float = 0.0,
    ) -> None:
        if not self._enabled:
            return
        if yaw_delta_rad is None or not math.isfinite(float(yaw_delta_rad)):
            return
        if frame_id <= self._last_frame_id:
            return
        with self._cam_lock:
            self._cam_yaw_cum += float(yaw_delta_rad)
            self._cam_samples.append(
                _CameraYawSample(
                    frame_id=int(frame_id),
                    yaw_cum_rad=float(self._cam_yaw_cum),
                    t_monotonic=float(t_monotonic),
                    motion_hint=float(abs(motion_hint)),
                )
            )
            self._last_frame_id = int(frame_id)

    def estimate_frame_time_usec(self, frame_id: int) -> int | None:
        if not self._enabled:
            return None
        with self._state_lock:
            if not self._state.ready:
                return None
            t_s = self._state.t_offset_s + (float(frame_id) / max(1e-6, self._state.fps_hz))
        return int(t_s * 1_000_000.0)

    def read_state(self) -> TimeSyncState:
        with self._state_lock:
            out = TimeSyncState(
                ready=self._state.ready,
                t_offset_s=self._state.t_offset_s,
                fps_hz=self._state.fps_hz,
                lag_s=self._state.lag_s,
                corr_peak=self._state.corr_peak,
                last_update_monotonic=self._state.last_update_monotonic,
                health_status=self._state.health_status,
                health_score=self._state.health_score,
                health_reason=self._state.health_reason,
                good_updates=self._state.good_updates,
                bad_updates=self._state.bad_updates,
            )
        if out.ready and out.last_update_monotonic > 0.0:
            age_s = time.monotonic() - out.last_update_monotonic
            if age_s > self._health_max_stale_s:
                out.health_status = "stale"
                out.health_reason = "no_recent_update"
                out.health_score = min(out.health_score, 0.2)
        return out

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass

    @staticmethod
    def _z_norm(x: Any, min_std: float) -> Any:
        import numpy as np  # type: ignore

        if x.size == 0:
            return None
        mu = float(np.mean(x))
        sigma = float(np.std(x))
        if sigma < min_std:
            return None
        return (x - mu) / sigma

    def _init_particles(self, cam: list[_CameraYawSample], imu: list[Any]) -> None:
        import numpy as np  # type: ignore

        if self._particles is not None and self._weights is not None:
            return
        if not cam or not imu:
            return

        frame0 = float(cam[0].frame_id)
        imu_start = float(imu[0].t_imu_s)
        # Model: t_img = t_offset + frame_id / fps.
        offset_center = imu_start - (frame0 / self._nominal_fps_hz)

        offsets = self._rng.normal(offset_center, self._init_offset_std_s, size=self._num_particles)
        fps = self._rng.normal(self._nominal_fps_hz, self._init_fps_std_hz, size=self._num_particles)
        fps = np.clip(fps, self._fps_min, self._fps_max)
        self._particles = np.column_stack((offsets, fps)).astype(np.float64)
        self._weights = np.full((self._num_particles,), 1.0 / float(self._num_particles), dtype=np.float64)

    def _current_best(self) -> tuple[float, float]:
        import numpy as np  # type: ignore

        if self._particles is None or self._weights is None:
            with self._state_lock:
                return self._state.t_offset_s, self._state.fps_hz
        idx = int(np.argmax(self._weights))
        return float(self._particles[idx, 0]), float(self._particles[idx, 1])

    def _systematic_resample(self) -> None:
        import numpy as np  # type: ignore

        if self._particles is None or self._weights is None:
            return
        cdf = np.cumsum(self._weights)
        if cdf.size == 0 or not math.isfinite(float(cdf[-1])) or float(cdf[-1]) <= 0.0:
            return
        cdf /= cdf[-1]
        step = 1.0 / float(self._num_particles)
        u0 = self._rng.uniform(0.0, step)
        u = u0 + step * np.arange(self._num_particles, dtype=np.float64)
        idx = np.searchsorted(cdf, u, side="right")
        idx = np.clip(idx, 0, self._num_particles - 1)
        self._particles = self._particles[idx].copy()
        self._weights.fill(1.0 / float(self._num_particles))

    def _reset_filter(self, *, reason: str, now_monotonic: float) -> None:
        self._particles = None
        self._weights = None
        self._corr_ema = 0.0
        self._lag_abs_ema_s = 0.0
        self._fps_err_ema = 0.0
        self._good_updates = 0
        self._bad_updates = 0
        self._health_has_sample = False
        with self._state_lock:
            self._state.ready = False
            self._state.lag_s = 0.0
            self._state.corr_peak = 0.0
            self._state.last_update_monotonic = now_monotonic
            self._state.health_status = "stale"
            self._state.health_score = min(self._state.health_score, 0.2)
            self._state.health_reason = reason
            self._state.good_updates = 0
            self._state.bad_updates = 0

    def _maybe_reset_on_stale(self, now_monotonic: float) -> None:
        with self._state_lock:
            ready = bool(self._state.ready)
            last_update = float(self._state.last_update_monotonic)
        if not ready or last_update <= 0.0:
            return
        if (now_monotonic - last_update) <= self._health_max_stale_s:
            return
        self._reset_filter(reason="rebootstrap", now_monotonic=now_monotonic)

    def _measurement_from_correlation(self, cam: list[_CameraYawSample], imu: list[Any]) -> tuple[float, int, float, float] | None:
        import numpy as np  # type: ignore

        if len(cam) < self._min_samples or len(imu) < self._min_samples:
            return None

        cam_ids = np.asarray([s.frame_id for s in cam], dtype=np.float64)
        cam_yaw = np.unwrap(np.asarray([s.yaw_cum_rad for s in cam], dtype=np.float64))
        imu_t = np.asarray([float(s.t_imu_s) for s in imu], dtype=np.float64)
        imu_yaw = np.unwrap(np.asarray([float(s.yaw_rad) for s in imu], dtype=np.float64))

        offset_s, fps_hz = self._current_best()
        cam_t_pred = offset_s + (cam_ids / max(1e-6, fps_hz))
        if cam_t_pred[-1] <= cam_t_pred[0]:
            return None

        t_min = float(cam_t_pred[0] - self._max_lag_s)
        t_max = float(cam_t_pred[-1] + self._max_lag_s)
        in_window = (imu_t >= t_min) & (imu_t <= t_max)
        if int(np.count_nonzero(in_window)) < self._min_samples:
            return None

        imu_t_w = imu_t[in_window]
        imu_yaw_w = imu_yaw[in_window]
        if imu_t_w.size < self._min_samples:
            return None

        cam_interp = np.interp(imu_t_w, cam_t_pred, cam_yaw, left=np.nan, right=np.nan)
        valid = np.isfinite(cam_interp) & np.isfinite(imu_yaw_w)
        if int(np.count_nonzero(valid)) < self._min_samples:
            return None
        cam_sig = cam_interp[valid]
        imu_sig = imu_yaw_w[valid]
        imu_t_sig = imu_t_w[valid]

        if cam_sig.size < self._min_samples:
            return None

        dts = np.diff(imu_t_sig)
        dts = dts[np.isfinite(dts) & (dts > 1e-6)]
        if dts.size == 0:
            return None
        dt = float(np.median(dts))
        if not math.isfinite(dt) or dt <= 0.0:
            return None

        imu_rate = np.diff(imu_sig) / np.maximum(np.diff(imu_t_sig), 1e-6)
        if imu_rate.size == 0:
            return None
        motion_rate = float(np.max(np.abs(imu_rate)))
        if motion_rate < self._min_angular_rate:
            return None

        cam_n = self._z_norm(cam_sig, self._min_signal_std)
        imu_n = self._z_norm(imu_sig, self._min_signal_std)
        if cam_n is None or imu_n is None:
            return None

        corr = np.correlate(cam_n, imu_n, mode="full")
        if corr.size == 0:
            return None
        best_idx = int(np.argmax(corr))
        lag_samples = best_idx - (cam_n.size - 1)
        lag_s = float(lag_samples) * dt
        if self._max_lag_s > 0.0 and abs(lag_s) > self._max_lag_s:
            return None

        corr_peak = float(corr[best_idx]) / float(cam_n.size)
        if corr_peak < self._min_corr_peak:
            return None

        frame_id = int(cam_ids[-1])
        t_pred = offset_s + (float(frame_id) / max(1e-6, fps_hz))
        # If camera sequence appears delayed (positive lag), frame time must shift earlier.
        z_t = float(t_pred - lag_s)
        return z_t, frame_id, lag_s, corr_peak

    def _update(self) -> None:
        import numpy as np  # type: ignore

        now_monotonic = time.monotonic()
        with self._cam_lock:
            cam = list(self._cam_samples)
        if len(cam) > self._cam_window:
            cam = cam[-self._cam_window :]
        if len(cam) < self._min_samples:
            self._maybe_reset_on_stale(now_monotonic)
            return

        imu = self._imu_source.read_yaw_samples(max_samples=self._imu_window, max_age_s=self._imu_max_age_s)
        if len(imu) < self._min_samples:
            self._maybe_reset_on_stale(now_monotonic)
            return

        self._init_particles(cam, imu)
        if self._particles is None or self._weights is None:
            return

        if self._proc_offset_noise_s > 0.0:
            self._particles[:, 0] += self._rng.normal(0.0, self._proc_offset_noise_s, size=self._num_particles)
        if self._proc_fps_noise_hz > 0.0:
            self._particles[:, 1] += self._rng.normal(0.0, self._proc_fps_noise_hz, size=self._num_particles)
        self._particles[:, 1] = np.clip(self._particles[:, 1], self._fps_min, self._fps_max)

        measurement = self._measurement_from_correlation(cam, imu)
        if measurement is None:
            self._maybe_reset_on_stale(now_monotonic)
            return
        z_t, frame_id, lag_s, corr_peak = measurement

        x_est = self._particles[:, 0] + (float(frame_id) / np.maximum(1e-6, self._particles[:, 1]))
        err = x_est - z_t
        coeff = 1.0 / (self._sigma_s * math.sqrt(2.0 * math.pi))
        like = coeff * np.exp(-0.5 * (err / self._sigma_s) ** 2)
        self._weights *= like + 1e-30

        w_sum = float(np.sum(self._weights))
        if not math.isfinite(w_sum) or w_sum <= 0.0:
            self._weights.fill(1.0 / float(self._num_particles))
            return
        self._weights /= w_sum

        best_idx = int(np.argmax(self._weights))
        best_offset = float(self._particles[best_idx, 0])
        best_fps = float(self._particles[best_idx, 1])
        health_status, health_score, health_reason = self._evaluate_health(
            best_fps=best_fps,
            lag_s=lag_s,
            corr_peak=corr_peak,
        )
        with self._state_lock:
            self._state.ready = True
            self._state.t_offset_s = best_offset
            self._state.fps_hz = best_fps
            self._state.lag_s = lag_s
            self._state.corr_peak = corr_peak
            self._state.last_update_monotonic = now_monotonic
            self._state.health_status = health_status
            self._state.health_score = health_score
            self._state.health_reason = health_reason
            self._state.good_updates = int(self._good_updates)
            self._state.bad_updates = int(self._bad_updates)

        self._systematic_resample()

    @staticmethod
    def _clamp01(v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    def _evaluate_health(self, *, best_fps: float, lag_s: float, corr_peak: float) -> tuple[str, float, str]:
        fps_err_rel = abs(float(best_fps) - self._nominal_fps_hz) / max(1e-6, self._nominal_fps_hz)
        lag_abs = abs(float(lag_s))
        corr = float(corr_peak)

        if not self._health_has_sample:
            self._corr_ema = corr
            self._lag_abs_ema_s = lag_abs
            self._fps_err_ema = fps_err_rel
            self._health_has_sample = True
        else:
            a = self._health_ema_alpha
            self._corr_ema += a * (corr - self._corr_ema)
            self._lag_abs_ema_s += a * (lag_abs - self._lag_abs_ema_s)
            self._fps_err_ema += a * (fps_err_rel - self._fps_err_ema)

        is_good = (
            corr >= self._health_min_corr
            and lag_abs <= self._health_max_abs_lag_s
            and fps_err_rel <= self._health_max_fps_error_rel
        )
        if is_good:
            self._good_updates += 1
            self._bad_updates = 0
        else:
            self._bad_updates += 1
            self._good_updates = 0

        corr_score = self._clamp01((self._corr_ema - self._health_min_corr) / max(1e-6, 1.0 - self._health_min_corr))
        lag_score = self._clamp01(1.0 - (self._lag_abs_ema_s / self._health_max_abs_lag_s))
        fps_score = self._clamp01(1.0 - (self._fps_err_ema / self._health_max_fps_error_rel))
        health_score = (0.5 * corr_score) + (0.3 * lag_score) + (0.2 * fps_score)
        health_score = self._clamp01(health_score)

        if self._good_updates < self._health_min_good_updates:
            return "warming_up", health_score, "stabilizing"
        if is_good:
            return "healthy", health_score, "ok"

        corr_def = max(0.0, (self._health_min_corr - self._corr_ema) / max(1e-6, self._health_min_corr))
        lag_def = max(0.0, (self._lag_abs_ema_s - self._health_max_abs_lag_s) / self._health_max_abs_lag_s)
        fps_def = max(0.0, (self._fps_err_ema - self._health_max_fps_error_rel) / self._health_max_fps_error_rel)

        reason = "unstable"
        if corr_def >= lag_def and corr_def >= fps_def and corr_def > 0.0:
            reason = "low_correlation"
        elif lag_def >= corr_def and lag_def >= fps_def and lag_def > 0.0:
            reason = "high_lag"
        elif fps_def > 0.0:
            reason = "fps_drift"
        return "degraded", health_score, reason

    def _loop(self) -> None:
        period = 1.0 / self._update_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._update()
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            sleep_s = period - elapsed
            if sleep_s > 0.0:
                self._stop.wait(timeout=sleep_s)
