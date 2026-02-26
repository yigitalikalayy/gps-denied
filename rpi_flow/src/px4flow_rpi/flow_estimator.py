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
        self._fb_check = bool(cfg.get("lk_fb_check", True))
        self._fb_thresh_px = float(cfg.get("lk_fb_thresh_px", 1.5))
        self._ransac_enable = bool(cfg.get("lk_ransac", True))
        self._ransac_thresh_px = float(cfg.get("lk_ransac_thresh_px", 2.0))
        self._ransac_confidence = float(cfg.get("lk_ransac_confidence", 0.99))
        self._ransac_max_iters = int(cfg.get("lk_ransac_max_iters", 2000))
        self._reseed_every = int(cfg.get("reseed_every_n_frames", 10))
        self._min_tracked = int(cfg.get("min_tracked_features", 10))
        self._quality_mode = str(cfg.get("quality_mode", "max_features")).strip().lower()
        self._min_inlier_ratio = float(cfg.get("min_inlier_ratio", 0.0))
        self._quality_mad_px = float(cfg.get("quality_mad_px", 0.0))
        self._quality_motion_px = float(cfg.get("quality_motion_px", 0.0))
        self._quality_motion_floor = float(cfg.get("quality_motion_floor", 0.0))
        self._quality_min = int(cfg.get("quality_min", 0))
        self._quality_max = int(cfg.get("quality_max", 255))
        self._flow_deadband_px = float(cfg.get("flow_deadband_px", 0.0))
        self._fallback_mode = str(cfg.get("fallback_mode", "none")).strip().lower()
        self._fallback_scale = float(cfg.get("fallback_scale", 1.0))
        self._fallback_min_response = float(cfg.get("fallback_min_response", 0.0))
        self._feature_detector = str(cfg.get("feature_detector", "shi_tomasi")).strip().lower()
        self._fast_threshold = int(cfg.get("fast_threshold", 20))
        self._fast_nonmax = bool(cfg.get("fast_nonmax_suppression", True))
        self._feature_border_px = int(cfg.get("feature_border_px", 8))
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
    def _robust_mean(dx: Any, dy: Any, k: float) -> tuple[float, float, int, float]:
        import numpy as np  # type: ignore

        if dx.size == 0:
            return 0.0, 0.0, 0, 0.0

        med_dx = float(np.median(dx))
        med_dy = float(np.median(dy))
        residual = np.hypot(dx - med_dx, dy - med_dy)
        mad = float(np.median(np.abs(residual))) + 1e-6
        keep = residual <= k * mad
        dx2 = dx[keep]
        dy2 = dy[keep]
        if dx2.size == 0:
            return med_dx, med_dy, int(dx.size), mad
        return float(np.mean(dx2)), float(np.mean(dy2)), int(dx2.size), mad

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
            cv2.FM_RANSAC,
            max(0.2, self._f_ransac_thresh_px),
            max(0.5, min(0.9999, self._f_confidence)),
            max(100, self._f_max_iters),
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
        if self._feature_border_px > 0:
            h, w = prev_p.shape[:2]
            b = int(self._feature_border_px)
            if h > 2 * b and w > 2 * b:
                border = np.zeros((h, w), dtype=np.uint8)
                border[b : h - b, b : w - b] = 255
                if mask is None:
                    mask = border
                else:
                    mask = cv2.bitwise_and(mask, border)
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

        if self._fb_check:
            p0b, st_b, _err_b = cv2.calcOpticalFlowPyrLK(gray_p, prev_p, p1g.reshape(-1, 1, 2), None, **lk_params)
            if p0b is not None and st_b is not None:
                back_ok = st_b.reshape(-1) == 1
                p0b = p0b.reshape(-1, 2)
                fb_err = np.linalg.norm(p0b - p0g, axis=1)
                fb_keep = back_ok & (fb_err <= max(0.1, self._fb_thresh_px))
                p0g = p0g[fb_keep]
                p1g = p1g[fb_keep]
                tracked = int(p0g.shape[0])
                if tracked < self._min_tracked:
                    self._p0 = None
                    return self._fallback_phase(prev_p, gray_p, requested)

        d = p1g - p0g
        dx = d[:, 0]
        dy = d[:, 1]

        inlier_mask = None
        if self._ransac_enable and tracked >= max(6, self._min_tracked):
            try:
                model, inliers = cv2.estimateAffinePartial2D(
                    p0g,
                    p1g,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=max(0.5, self._ransac_thresh_px),
                    confidence=max(0.5, min(0.9999, self._ransac_confidence)),
                    maxIters=max(200, self._ransac_max_iters),
                )
            except Exception:
                model, inliers = None, None
            if inliers is not None:
                inlier_mask = inliers.reshape(-1).astype(bool)
                if int(np.count_nonzero(inlier_mask)) < self._min_tracked:
                    inlier_mask = None

        if inlier_mask is not None:
            dx_in = dx[inlier_mask]
            dy_in = dy[inlier_mask]
            dx_m, dy_m, _kept, mad = self._robust_mean(dx_in, dy_in, self._mad_k)
            residual = np.hypot(dx_in - dx_m, dy_in - dy_m)
            keep = residual <= (self._mad_k * max(1e-6, mad))
            p0_use = p0g[inlier_mask][keep]
            p1_use = p1g[inlier_mask][keep]
        else:
            dx_m, dy_m, _kept, mad = self._robust_mean(dx, dy, self._mad_k)
            residual = np.hypot(dx - dx_m, dy - dy_m)
            keep = residual <= (self._mad_k * max(1e-6, mad))
            p0_use = p0g[keep]
            p1_use = p1g[keep]
        kept = int(np.count_nonzero(keep))
        if kept < self._min_tracked:
            self._p0 = None
            return self._fallback_phase(prev_p, gray_p, requested)

        if self._quality_mode == "requested":
            denom = max(1, requested)
        else:
            denom = max(1, self._max_features)
        base_quality = max(0.0, min(1.0, kept / denom))
        if self._quality_mad_px > 0.0 and mad > self._quality_mad_px:
            base_quality *= max(0.0, min(1.0, self._quality_mad_px / mad))
        motion_mag = float(math.hypot(dx_m, dy_m))
        if self._quality_motion_px > 0.0:
            motion_scale = self._quality_motion_px / max(self._quality_motion_px, motion_mag)
            motion_scale = max(self._quality_motion_floor, motion_scale)
            base_quality *= max(0.0, min(1.0, motion_scale))
        if self._min_inlier_ratio > 0.0 and (kept / denom) < self._min_inlier_ratio:
            base_quality = 0.0
        quality = int(max(self._quality_min, min(self._quality_max, base_quality * 255.0)))

        yaw_rad = None
        geom_inliers = 0
        if self._k is not None and p0_use.shape[0] >= self._min_geom_matches:
            yaw_rad, geom_inliers = self._estimate_yaw_from_geometry(p0_use, p1_use)

        if self._flow_deadband_px > 0.0 and motion_mag < self._flow_deadband_px:
            dx_m = 0.0
            dy_m = 0.0
            motion_mag = 0.0

        self._p0 = p1_use.reshape(-1, 1, 2).astype(np.float32, copy=False)
        return FlowResult(
            dx_px=dx_m,
            dy_px=dy_m,
            quality_u8=quality,
            tracked=kept,
            requested=requested,
            yaw_rad=yaw_rad,
            geom_inliers=geom_inliers,
            motion_mag=motion_mag,
        )


class Px4FlowEstimator:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._image_size = int(cfg.get("px4flow_image_size", 64))
        self._search_size = int(cfg.get("px4flow_search_size", 4))
        self._feature_threshold = int(cfg.get("px4flow_feature_threshold", 40))
        self._value_threshold = int(cfg.get("px4flow_value_threshold", 2000))
        self._tile_size = int(cfg.get("px4flow_tile_size", 8))
        self._num_blocks = int(cfg.get("px4flow_num_blocks", 5))
        self._min_count = int(cfg.get("px4flow_min_count", 10))
        self._flow_deadband_px = float(cfg.get("flow_deadband_px", 0.6))
        self._flow_edge_reject_margin_px = float(cfg.get("flow_edge_reject_margin_px", 0.25))
        self._quality_mad_px = float(cfg.get("quality_mad_px", 0.6))
        self._quality_motion_px = float(cfg.get("quality_motion_px", 0.4))
        self._quality_motion_floor = float(cfg.get("quality_motion_floor", 0.2))
        self._quality_min = int(cfg.get("quality_min", 0))
        self._quality_max = int(cfg.get("quality_max", 255))
        outlier_mad = cfg.get("px4flow_outlier_mad", None)
        if outlier_mad is None:
            outlier_mad = cfg.get("outlier_reject_mad", 4.0)
        try:
            self._outlier_mad = float(outlier_mad)
        except Exception:
            self._outlier_mad = 4.0
        if self._image_size < self._tile_size * 2:
            self._image_size = self._tile_size * 2
        if self._search_size < 1:
            self._search_size = 1
        if self._num_blocks < 1:
            self._num_blocks = 1
        if self._tile_size < 4:
            self._tile_size = 4
        self._half_tile = (self._tile_size - 1) * 0.5

    def set_camera_intrinsics(self, _camera_matrix: Any | None) -> None:
        return None

    def _prepare(self, gray: Any) -> Any:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if gray is None:
            return None
        if gray.dtype != np.uint8:
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        else:
            gray_u8 = gray
        if gray_u8.shape[0] != self._image_size or gray_u8.shape[1] != self._image_size:
            gray_u8 = cv2.resize(gray_u8, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA)
        return gray_u8

    def _compute_diff(self, img: Any, off_x: int, off_y: int) -> int:
        import numpy as np  # type: ignore

        x0 = off_x + 2
        y0 = off_y + 2
        patch = img[y0 : y0 + 4, x0 : x0 + 4]
        if patch.shape != (4, 4):
            return 0
        row_diff = np.abs(patch[0] - patch[1]).sum()
        row_diff += np.abs(patch[1] - patch[2]).sum()
        row_diff += np.abs(patch[2] - patch[3]).sum()
        col_diff = np.abs(patch[:, 0] - patch[:, 1]).sum()
        col_diff += np.abs(patch[:, 1] - patch[:, 2]).sum()
        col_diff += np.abs(patch[:, 2] - patch[:, 3]).sum()
        return int(row_diff + col_diff)

    def _sad_8x8(self, img1: Any, img2: Any, x1: int, y1: int, x2: int, y2: int) -> int:
        import numpy as np  # type: ignore

        p1 = img1[y1 : y1 + self._tile_size, x1 : x1 + self._tile_size]
        p2 = img2[y2 : y2 + self._tile_size, x2 : x2 + self._tile_size]
        if p1.shape != (self._tile_size, self._tile_size) or p2.shape != (self._tile_size, self._tile_size):
            return 1_000_000_000
        return int(np.abs(p1.astype(np.int16) - p2.astype(np.int16)).sum())

    def _subpixel_dir(self, img1: Any, img2: Any, x: int, y: int, dx: int, dy: int) -> tuple[float, float]:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        patch1 = img1[y : y + self._tile_size, x : x + self._tile_size]
        if patch1.shape != (self._tile_size, self._tile_size):
            return 0.0, 0.0
        patch1 = patch1.astype(np.float32)
        cx = float(x + dx) + self._half_tile
        cy = float(y + dy) + self._half_tile
        dirs = [
            (0.5, 0.0),
            (0.5, 0.5),
            (0.0, 0.5),
            (-0.5, 0.5),
            (-0.5, 0.0),
            (-0.5, -0.5),
            (0.0, -0.5),
            (0.5, -0.5),
        ]
        best = None
        best_dir = (0.0, 0.0)
        for off_x, off_y in dirs:
            center = (cx + off_x, cy + off_y)
            patch2 = cv2.getRectSubPix(img2, (self._tile_size, self._tile_size), center)
            if patch2 is None or patch2.shape != (self._tile_size, self._tile_size):
                continue
            sad = float(np.abs(patch1 - patch2).sum())
            if best is None or sad < best:
                best = sad
                best_dir = (off_x, off_y)
        return best_dir

    def estimate(self, prev_gray: Any, gray: Any) -> FlowResult:
        import math

        prev = self._prepare(prev_gray)
        cur = self._prepare(gray)
        if prev is None or cur is None:
            return FlowResult(dx_px=0.0, dy_px=0.0, quality_u8=0, tracked=0, requested=0)

        img_w = self._image_size
        winmin = -self._search_size
        winmax = self._search_size
        pix_lo = self._search_size + 1
        pix_hi = img_w - (self._search_size + 1) - self._tile_size
        pix_step = (pix_hi - pix_lo) // self._num_blocks + 1 if pix_hi > pix_lo else 1

        meancount = 0
        sum_x = 0.0
        sum_y = 0.0
        samples_dx: list[float] = []
        samples_dy: list[float] = []

        for y in range(pix_lo, pix_hi + 1, pix_step):
            for x in range(pix_lo, pix_hi + 1, pix_step):
                diff = self._compute_diff(prev, x, y)
                if diff < self._feature_threshold:
                    continue

                best = None
                best_dx = 0
                best_dy = 0
                for dy in range(winmin, winmax + 1):
                    for dx in range(winmin, winmax + 1):
                        sad = self._sad_8x8(prev, cur, x, y, x + dx, y + dy)
                        if best is None or sad < best:
                            best = sad
                            best_dx = dx
                            best_dy = dy

                if best is None or best >= self._value_threshold:
                    continue

                sub_dx, sub_dy = self._subpixel_dir(prev, cur, x, y, best_dx, best_dy)
                dx_f = float(best_dx) + float(sub_dx)
                dy_f = float(best_dy) + float(sub_dy)
                samples_dx.append(dx_f)
                samples_dy.append(dy_f)
                sum_x += dx_f
                sum_y += dy_f
                meancount += 1

        requested = self._num_blocks * self._num_blocks
        if meancount > self._min_count:
            # Robust outlier rejection using median absolute deviation.
            if self._outlier_mad > 0.0 and meancount >= max(6, self._min_count):
                try:
                    import numpy as np  # type: ignore

                    dx_arr = np.asarray(samples_dx, dtype=np.float64)
                    dy_arr = np.asarray(samples_dy, dtype=np.float64)
                    med_x = float(np.median(dx_arr))
                    med_y = float(np.median(dy_arr))
                    dev = np.hypot(dx_arr - med_x, dy_arr - med_y)
                    mad = float(np.median(dev))
                    if mad > 1e-6:
                        keep = dev <= (self._outlier_mad * mad)
                        if int(np.count_nonzero(keep)) >= self._min_count:
                            dx_arr = dx_arr[keep]
                            dy_arr = dy_arr[keep]
                            meancount = int(dx_arr.size)
                            sum_x = float(dx_arr.sum())
                            sum_y = float(dy_arr.sum())
                except Exception:
                    pass
            dx = sum_x / float(meancount)
            dy = sum_y / float(meancount)
            if self._flow_deadband_px > 0.0:
                if abs(dx) < self._flow_deadband_px:
                    dx = 0.0
                if abs(dy) < self._flow_deadband_px:
                    dy = 0.0
            quality = int(max(0, min(255, (meancount * 255) // max(1, requested))))
            try:
                import numpy as np  # type: ignore

                if samples_dx and self._quality_mad_px > 0.0:
                    dx_arr = np.asarray(samples_dx, dtype=np.float64)
                    dy_arr = np.asarray(samples_dy, dtype=np.float64)
                    med_x = float(np.median(dx_arr))
                    med_y = float(np.median(dy_arr))
                    dev = np.hypot(dx_arr - med_x, dy_arr - med_y)
                    mad = float(np.median(dev))
                    if math.isfinite(mad) and mad > 1e-6:
                        ratio = mad / max(1e-6, self._quality_mad_px)
                        coherence = 1.0 / (1.0 + (ratio * ratio))
                        quality = int(round(float(quality) * coherence))
                if self._quality_motion_px > 0.0:
                    motion_mag = math.hypot(float(dx), float(dy))
                    motion_score = min(1.0, max(0.0, motion_mag / self._quality_motion_px))
                    motion_floor = max(0.0, min(1.0, self._quality_motion_floor))
                    motion_score = max(motion_floor, motion_score)
                    quality = int(round(float(quality) * motion_score))
            except Exception:
                pass
            if self._flow_edge_reject_margin_px >= 0.0:
                edge = float(self._search_size) + float(self._flow_edge_reject_margin_px)
                if abs(dx) >= edge or abs(dy) >= edge:
                    dx = 0.0
                    dy = 0.0
                    quality = 0
            quality = int(max(self._quality_min, min(self._quality_max, quality)))
        else:
            dx = 0.0
            dy = 0.0
            quality = 0

        return FlowResult(
            dx_px=float(dx),
            dy_px=float(dy),
            quality_u8=int(quality),
            tracked=int(meancount),
            requested=int(requested),
            motion_mag=float(math.hypot(float(dx), float(dy))),
        )
