from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FlowResult:
    dx_px: float
    dy_px: float
    quality_u8: int
    tracked: int
    requested: int


class OpticalFlowEstimator:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._max_features = int(cfg.get("feature_max", 200))
        self._feature_quality = float(cfg.get("feature_quality", 0.01))
        self._min_distance = int(cfg.get("feature_min_distance", 12))
        self._win_size = int(cfg.get("lk_win_size", 21))
        self._max_level = int(cfg.get("lk_max_level", 3))
        self._mad_k = float(cfg.get("outlier_reject_mad", 3.5))
        self._reseed_every = int(cfg.get("reseed_every_n_frames", 10))
        self._min_tracked = int(cfg.get("min_tracked_features", 30))

        self._frame_idx = 0
        self._p0 = None

    def _detect(self, gray: Any) -> Any:
        import cv2  # type: ignore

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_features,
            qualityLevel=self._feature_quality,
            minDistance=self._min_distance,
            blockSize=7,
            useHarrisDetector=False,
        )

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

    def estimate(self, prev_gray: Any, gray: Any) -> FlowResult:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        self._frame_idx += 1
        reseed = (self._p0 is None) or (self._frame_idx % self._reseed_every == 0)
        if reseed:
            self._p0 = self._detect(prev_gray)

        requested = 0 if self._p0 is None else int(self._p0.shape[0])
        if self._p0 is None or requested == 0:
            return FlowResult(dx_px=0.0, dy_px=0.0, quality_u8=0, tracked=0, requested=0)

        lk_params = dict(
            winSize=(self._win_size, self._win_size),
            maxLevel=self._max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        p1, st, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self._p0, None, **lk_params)
        if p1 is None or st is None:
            self._p0 = None
            return FlowResult(dx_px=0.0, dy_px=0.0, quality_u8=0, tracked=0, requested=requested)

        good = st.reshape(-1) == 1
        p0g = self._p0.reshape(-1, 2)[good]
        p1g = p1.reshape(-1, 2)[good]
        tracked = int(p0g.shape[0])

        if tracked < self._min_tracked:
            self._p0 = None
            return FlowResult(dx_px=0.0, dy_px=0.0, quality_u8=0, tracked=tracked, requested=requested)

        d = p1g - p0g
        dx = d[:, 0]
        dy = d[:, 1]

        dx_m, dy_m, kept = self._robust_mean(dx, dy, self._mad_k)
        quality = int(max(0.0, min(1.0, kept / max(1, self._max_features))) * 255.0)

        self._p0 = p1g.reshape(-1, 1, 2).astype(np.float32, copy=False)
        return FlowResult(dx_px=dx_m, dy_px=dy_m, quality_u8=quality, tracked=kept, requested=requested)

