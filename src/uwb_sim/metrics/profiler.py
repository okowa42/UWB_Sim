from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional


@dataclass
class TimerStats:
    n: int = 0
    total_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0

    def add(self, dt_s: float) -> None:
        self.n += 1
        self.total_s += dt_s
        self.min_s = min(self.min_s, dt_s)
        self.max_s = max(self.max_s, dt_s)

    @property
    def mean_s(self) -> float:
        return self.total_s / self.n if self.n > 0 else 0.0


class Timer:
    """
    Simple manual timer:
      t = Timer()
      t.start()
      ...work...
      dt = t.stop()
    """
    def __init__(self) -> None:
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Timer.stop() called before start().")
        dt = time.perf_counter() - self._t0
        self._t0 = None
        return dt
