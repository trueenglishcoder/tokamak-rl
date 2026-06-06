from __future__ import annotations

from dataclasses import dataclass, field
import sys
import time
from typing import TextIO


@dataclass(slots=True)
class TrainingProgressBar:
    """Small dependency-free progress bar for trainer CLI runs."""

    total_steps: int
    label: str = "training"
    enabled: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    width: int = 28
    min_interval_s: float = 0.25
    _started_at: float = field(default_factory=time.perf_counter, init=False)
    _last_render_at: float = field(default=0.0, init=False)
    _last_step: int = field(default=0, init=False)
    _last_rendered_step: int = field(default=-1, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if int(self.total_steps) <= 0:
            raise ValueError("total_steps must be > 0")
        if int(self.width) <= 0:
            raise ValueError("width must be > 0")
        if float(self.min_interval_s) < 0.0:
            raise ValueError("min_interval_s must be >= 0")
        self.total_steps = int(self.total_steps)
        self.width = int(self.width)
        self.min_interval_s = float(self.min_interval_s)

    def update(self, step: int, *, status: dict[str, object] | None = None, force: bool = False) -> None:
        if not self.enabled or self._closed:
            return
        current_step = max(0, min(int(step), self.total_steps))
        now = time.perf_counter()
        if (not force) and current_step < self.total_steps and now - self._last_render_at < self.min_interval_s:
            self._last_step = current_step
            return
        self._last_step = current_step
        self._last_render_at = now
        text = self._render(current_step, now=now, status=status or {})
        if _is_interactive(self.stream):
            self.stream.write("\r" + text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()
        self._last_rendered_step = current_step

    def close(self, *, status: dict[str, object] | None = None) -> None:
        if not self.enabled or self._closed:
            return
        if self._last_rendered_step != self._last_step or self._last_step < self.total_steps:
            self.update(self._last_step, status=status, force=True)
        if _is_interactive(self.stream):
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def _render(self, step: int, *, now: float, status: dict[str, object]) -> str:
        elapsed = max(now - self._started_at, 0.0)
        ratio = float(step) / float(self.total_steps)
        filled = int(round(ratio * float(self.width)))
        bar = "#" * filled + "-" * (self.width - filled)
        rate = float(step) / elapsed if elapsed > 0.0 and step > 0 else 0.0
        remaining = max(self.total_steps - int(step), 0)
        eta = remaining / rate if rate > 0.0 else None
        parts = [
            f"{self.label}",
            f"[{bar}]",
            f"{step}/{self.total_steps}",
            f"{ratio * 100.0:5.1f}%",
            f"{rate:6.2f} step/s",
            f"elapsed {_format_duration(elapsed)}",
            f"eta {_format_duration(eta)}",
        ]
        extra = _format_status(status)
        if extra:
            parts.append(extra)
        return " ".join(parts)


def _format_status(status: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in status.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.5g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(int(round(float(seconds))), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _is_interactive(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty()) if isatty is not None else False
    except OSError:
        return False
