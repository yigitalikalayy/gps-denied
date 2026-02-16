from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class FlowResult:
    dx_px: float
    dy_px: float
    quality_u8: int
    tracked: int
    requested: int
    yaw_rad: float | None = None
    geom_inliers: int = 0
    motion_mag: float = 0.0


class OpticalFlowEstimator:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._max_features = int(cfg.get("feature_max", 500))
        self._feature_quality = float(cfg.get("feature_quality", 0.01))
        self._min_distance = int(cfg.get("feature_min_distance", 12))
        self._win_size = int(cfg.get("lk_win_size", 21))
        self._max_level = int(cfg.get("lk_max_level", 3))
        self._mad_k = float(cfg.get("outlier_reject_mad", 3.5))
        self._reseed_every = int(cfg.get("reseed_every_n_frames", 10))
        self._min_tracked = int(cfg.get("min_tracked_features", 10))
        self._quality_mode = str(cfg.get("quality_mode", "max_features")).strip().lower()
        self._fallback_mode = str(cfg.get("fallback_mode", "none")).strip().lower()
        self._fallback_scale = float(cfg.get("fallback_scale", 1.0))
        self._fallback_min_response = float(cfg.get("fallback_min_response", 0.0))
        self._feature_detector = str(cfg.get("feature_detector", "shi_tomasi")).strip().lower()
        self._fast_threshold = int(cfg.get("fast_threshold", 20))
        self._fast_nonmax = bool(cfg.get("fast_nonmax_suppression", True))
        self._min_geom_matches = max(5, int(cfg.get("min_geom_matches", 8)))
        self._f_ransac_thresh_px = float(cfg.get("fundamental_ransac_thresh_px", 1.5))
        self._f_confidence = float(cfg.get("fundamental_confidence", 0.99))
        self._f_max_iters = int(cfg.get("fundamental_max_iters", 2000))
        self._e_ransac_thresh_px = float(cfg.get("essential_ransac_thresh_px", 1.0))
        self._e_confidence = float(cfg.get("essential_confidence", 0.999))
        pre_cfg = cfg.get("preprocess", {}) if isinstance(cfg.get("preprocess", {}), dict) else {}
        self._clahe_clip = float(pre_cfg.get("clahe_clip", 0.0))
        self._clahe_tile = int(pre_cfg.get("clahe_tile", 8))
        self._gamma = float(pre_cfg.get("gamma", 1.0))
        self._saturation_thresh = int(pre_cfg.get("saturation_threshold", 0))

        self._frame_idx = 0
        self._p0 = None
        self._clahe = None
        self._gamma_lut = None
        self._pc_win = None
        self._fast = None
        self._k = None

    def set_camera_intrinsics(self, camera_matrix: Any | None) -> None:
        import numpy as np  # type: ignore

        if camera_matrix is None:
            self._k = None
            return
        k = np.asarray(camera_matrix, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError(f"camera_matrix must be 3x3, got {k.shape}")
        self._k = k

    def _detect(self, gray: Any, mask: Any | None = None) -> Any:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if self._feature_detector == "fast":
            if self._fast is None:
                self._fast = cv2.FastFeatureDetector_create(
                    threshold=max(1, self._fast_threshold),
                    nonmaxSuppression=bool(self._fast_nonmax),
                )
            kps = self._fast.detect(gray, mask=mask)
            if not kps:
                return None
            kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[: self._max_features]
            pts = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
            return pts

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_features,
            qualityLevel=self._feature_quality,
            minDistance=self._min_distance,
            mask=mask,
            blockSize=7,
            useHarrisDetector=False,
        )

    def _preprocess(self, gray: Any) -> Any:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        out = gray
        if self._clahe_clip > 0.0:
            if self._clahe is None:
                tile = max(2, int(self._clahe_tile))
                self._clahe = cv2.createCLAHE(clipLimit=max(0.1, self._clahe_clip), tileGridSize=(tile, tile))
            out = self._clahe.apply(out)

        if self._gamma and abs(self._gamma - 1.0) > 1e-3:
            if self._gamma_lut is None or len(self._gamma_lut) != 256:
                inv = 1.0 / max(1e-6, self._gamma)
                self._gamma_lut = np.array(
                    [min(255, int(((i / 255.0) ** inv) * 255.0 + 0.5)) for i in range(256)], dtype=np.uint8
                )
            out = cv2.LUT(out, self._gamma_lut)

        return out

    @staticmethod
    def _robust_mean(dx: Any, dy: Any, k: float) -> tuple[float, float, int]:
        import numpy as np  # type: ignore

        if dx.size == 0:
            return 0.0, 0.0, 0

        med_dx = float(np.median(dx))
        med_dy = float(np.median(dy))
        mad_dx = float(np.median(np.abs(dx - med_dx))) + 1e-6
        mad_dy = float(np.median(np.abs(dy - med_dy))) + 1e-6

        keep = (np.abs(dx - med_dx) <= k * mad_dx) & (np.abs(dy - med_dy) <= k * mad_dy)
        dx2 = dx[keep]
        dy2 = dy[keep]
        if dx2.size == 0:
            return med_dx, med_dy, int(dx.size)
        return float(np.mean(dx2)), float(np.mean(dy2)), int(dx2.size)

    def _fallback_phase(self, prev_gray: Any, gray: Any, requested: int) -> FlowResult:
        if self._fallback_mode != "phase":
            return FlowResult(dx_px=0.0, dy_px=0.0, quality_u8=0, tracked=0, requested=requested)

        import cv2  # type: ignore
        import numpy as np  # type: ignore

        scale = max(0.1, min(1.0, self._fallback_scale))
        if scale < 0.999:
            prev_s = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray_s = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            prev_s = prev_gray
            gray_s = gray

        prev_f = np.float32(prev_s)
        gray_f = np.float32(gray_s)

        # Hanning window improves phase correlation stability under vibration.
        if self._pc_win is None or self._pc_win.shape != prev_f.shape:
            h, w = prev_f.shape[:2]
            self._pc_win = cv2.createHanningWindow((w, h), cv2.CV_32F)

        try:
            (shift_x, shift_y), response = cv2.phaseCorrelate(prev_f, gray_f, self._pc_win)
        except Exception:
            return FlowResult(dx_px=0.0, dy_px=0.0, quality_u8=0, tracked=0, requested=requested)

        if scale != 1.0:
            shift_x /= scale
            shift_y /= scale

        resp = float(response)
        if resp < self._fallback_min_response:
            quality = 0
        else:
            quality = int(max(0.0, min(1.0, resp)) * 255.0)

        return FlowResult(
            dx_px=float(shift_x),
            dy_px=float(shift_y),
            quality_u8=quality,
            tracked=0,
            requested=requested,
            motion_mag=float(math.hypot(float(shift_x), float(shift_y))),
        )

    @staticmethod
    def _rotation_to_euler_zyx(r: Any) -> tuple[float, float, float]:
        sy = math.sqrt(float(r[0, 0]) * float(r[0, 0]) + float(r[1, 0]) * float(r[1, 0]))
        singular = sy < 1e-8
        if not singular:
            roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
            pitch = math.atan2(-float(r[2, 0]), sy)
            yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
        else:
            roll = math.atan2(-float(r[1, 2]), float(r[1, 1]))
            pitch = math.atan2(-float(r[2, 0]), sy)
            yaw = 0.0
        return roll, pitch, yaw

    @staticmethod
    def _iter_essential_candidates(e: Any) -> list[Any]:
        import numpy as np  # type: ignore

        if e is None:
            return []
        arr = np.asarray(e, dtype=np.float64)
        if arr.shape == (3, 3):
            return [arr]
        if arr.ndim == 2 and arr.shape[1] == 3 and (arr.shape[0] % 3) == 0:
            return [arr[i : i + 3, :] for i in range(0, arr.shape[0], 3)]
        return []

    def _estimate_yaw_from_geometry(self, p0: Any, p1: Any) -> tuple[float | None, int]:
        if self._k is None or p0 is None or p1 is None:
            return None, 0

        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if int(p0.shape[0]) < self._min_geom_matches or int(p1.shape[0]) < self._min_geom_matches:
            return None, 0

        f_mat, f_mask = cv2.findFundamentalMat(
            p0,
            p1,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=max(0.2, self._f_ransac_thresh_px),
            confidence=max(0.5, min(0.9999, self._f_confidence)),
            maxIters=max(100, self._f_max_iters),
        )
        if f_mat is None or f_mask is None:
            return None, 0

        f_arr = np.asarray(f_mat, dtype=np.float64)
        if f_arr.shape != (3, 3):
            if f_arr.ndim == 2 and f_arr.shape[1] == 3 and (f_arr.shape[0] % 3) == 0:
                f_arr = f_arr[:3, :]
            else:
                return None, 0

        inlier_mask = f_mask.reshape(-1).astype(bool)
        inliers = int(np.count_nonzero(inlier_mask))
        if inliers < self._min_geom_matches:
            return None, inliers

        p0_in = np.asarray(p0[inlier_mask], dtype=np.float64)
        p1_in = np.asarray(p1[inlier_mask], dtype=np.float64)
        if p0_in.shape[0] < self._min_geom_matches:
            return None, int(p0_in.shape[0])

        # Required by the pipeline: E = K^T F K.
        e_from_f = self._k.T @ f_arr @ self._k

        # Also run OpenCV's 5-point estimation (Nister family) and prefer it if valid.
        e_5pt, _e_mask = cv2.findEssentialMat(
            p0_in,
            p1_in,
            self._k,
            method=cv2.RANSAC,
            prob=max(0.5, min(0.9999, self._e_confidence)),
            threshold=max(0.2, self._e_ransac_thresh_px),
        )

        candidates = self._iter_essential_candidates(e_5pt)
        if not candidates:
            candidates = [e_from_f]
        else:
            candidates.append(e_from_f)

        best_yaw = None
        best_pose_inliers = 0
        for e_candidate in candidates:
            try:
                pose_inliers, r_mat, _t_vec, _pose_mask = cv2.recoverPose(e_candidate, p0_in, p1_in, self._k)
            except Exception:
                continue
            pose_inliers_i = int(pose_inliers)
            if pose_inliers_i <= 0:
                continue
            if pose_inliers_i > best_pose_inliers:
                _roll, _pitch, yaw = self._rotation_to_euler_zyx(r_mat)
                best_yaw = float(yaw)
                best_pose_inliers = pose_inliers_i

        return best_yaw, best_pose_inliers

    def estimate(self, prev_gray: Any, gray: Any) -> FlowResult:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        self._frame_idx += 1
        prev_p = self._preprocess(prev_gray)
        gray_p = self._preprocess(gray)
        mask = None
        if self._saturation_thresh > 0:
            mask = (prev_p < self._saturation_thresh).astype(np.uint8) * 255
        reseed = (self._p0 is None) or (self._frame_idx % self._reseed_every == 0)
        if reseed:
            self._p0 = self._detect(prev_p, mask)

        requested = 0 if self._p0 is None else int(self._p0.shape[0])
        if self._p0 is None or requested == 0:
            return self._fallback_phase(prev_p, gray_p, requested)

        lk_params = dict(
            winSize=(self._win_size, self._win_size),
            maxLevel=self._max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_p, gray_p, self._p0, None, **lk_params)
        if p1 is None or st is None:
            self._p0 = None
            return self._fallback_phase(prev_p, gray_p, requested)

        good = st.reshape(-1) == 1
        p0g = self._p0.reshape(-1, 2)[good]
        p1g = p1.reshape(-1, 2)[good]
        tracked = int(p0g.shape[0])

        if tracked < self._min_tracked:
            self._p0 = None
            return self._fallback_phase(prev_p, gray_p, requested)

        d = p1g - p0g
        dx = d[:, 0]
        dy = d[:, 1]

        dx_m, dy_m, kept = self._robust_mean(dx, dy, self._mad_k)
        if self._quality_mode == "requested":
            denom = max(1, requested)
        else:
            denom = max(1, self._max_features)
        quality = int(max(0.0, min(1.0, kept / denom)) * 255.0)

        yaw_rad = None
        geom_inliers = 0
        if self._k is not None and tracked >= self._min_geom_matches:
            yaw_rad, geom_inliers = self._estimate_yaw_from_geometry(p0g, p1g)

        self._p0 = p1g.reshape(-1, 1, 2).astype(np.float32, copy=False)
        return FlowResult(
            dx_px=dx_m,
            dy_px=dy_m,
            quality_u8=quality,
            tracked=kept,
            requested=requested,
            yaw_rad=yaw_rad,
            geom_inliers=geom_inliers,
            motion_mag=float(math.hypot(dx_m, dy_m)),
        )
