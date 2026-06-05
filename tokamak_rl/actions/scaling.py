from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ActionScaler:
    """Map normalized policy actions to physical current derivatives."""

    derivative_scale: np.ndarray
    fallback_derivative_scale: float | np.ndarray | None = None

    def __post_init__(self) -> None:
        scale = np.asarray(self.derivative_scale, dtype=float).reshape(-1)
        if scale.size == 0:
            raise ValueError("derivative_scale must not be empty")
        invalid = (~np.isfinite(scale)) | (scale <= 0.0)
        if np.any(invalid):
            if self.fallback_derivative_scale is None:
                raise ValueError("invalid derivative_scale entries require explicit fallback_derivative_scale")
            fallback = _coerce_fallback(self.fallback_derivative_scale, shape=scale.shape)
            scale = scale.copy()
            scale[invalid] = fallback[invalid]
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("derivative_scale must contain positive finite values")
        object.__setattr__(self, "derivative_scale", scale.copy())

    @property
    def action_dim(self) -> int:
        return int(self.derivative_scale.size)

    def to_physical(self, action_norm: np.ndarray) -> np.ndarray:
        action = np.asarray(action_norm, dtype=float).reshape(-1)
        if action.shape != self.derivative_scale.shape:
            raise ValueError(f"action shape {action.shape} != {self.derivative_scale.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain finite values")
        return np.clip(action, -1.0, 1.0) * self.derivative_scale

    def to_normalized(self, physical_derivatives: np.ndarray) -> np.ndarray:
        derivatives = np.asarray(physical_derivatives, dtype=float).reshape(-1)
        if derivatives.shape != self.derivative_scale.shape:
            raise ValueError(f"derivative shape {derivatives.shape} != {self.derivative_scale.shape}")
        if not np.all(np.isfinite(derivatives)):
            raise ValueError("derivatives must contain finite values")
        return np.clip(derivatives / self.derivative_scale, -1.0, 1.0)


def _coerce_fallback(value: float | np.ndarray, *, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(shape, float(arr), dtype=float)
    else:
        arr = arr.reshape(-1)
        if arr.shape != shape:
            raise ValueError(f"fallback_derivative_scale shape {arr.shape} != {shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("fallback_derivative_scale must contain positive finite values")
    return arr
