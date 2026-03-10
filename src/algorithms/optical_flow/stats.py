from __future__ import annotations

import time


class Rate:
    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = float(alpha)
        self._last_t = None
        self._hz = 0.0

    def tick(self, t: float | None = None) -> float:
        if t is None:
            t = time.monotonic()
        if self._last_t is None:
            self._last_t = t
            return self._hz
        dt = max(1e-6, t - self._last_t)
        inst = 1.0 / dt
        if self._hz <= 0.0:
            self._hz = inst
        else:
            self._hz = (1.0 - self._alpha) * self._hz + self._alpha * inst
        self._last_t = t
        return self._hz

