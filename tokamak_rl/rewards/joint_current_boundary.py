from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class JointCurrentBoundaryReward:
    """Reward component calculator for Ip and target-point boundary tracking."""

    ip_weight: float = 1.0
    shape_weight: float = 1.0
    action_weight: float = 0.01
    delta_action_weight: float = 0.01
    current_limit_weight: float = 0.0
    derivative_limit_weight: float = 0.0
    termination_penalty: float = 10.0
    ip_tolerance_norm: float = 0.05
    shape_tolerance_norm: float = 0.02

    def __call__(
        self,
        *,
        true_ip: float,
        ip_ref: float,
        ip_scale: float,
        true_boundary_poly: np.ndarray | None,
        reference_boundary_points: np.ndarray,
        radius_scale: float,
        action_norm: np.ndarray,
        previous_action_norm: np.ndarray,
        terminated: bool,
        current_limit_margin: np.ndarray | None = None,
        derivative_limit_margin: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute a scalar reward and inspectable component values."""
        ip_error_norm = abs(float(true_ip) - float(ip_ref)) / max(float(ip_scale), 1.0)
        r_ip = float(np.exp(-((ip_error_norm / max(float(self.ip_tolerance_norm), 1e-12)) ** 2)))
        reference_points = _as_points(reference_boundary_points, "reference_boundary_points")
        if true_boundary_poly is None:
            shape_error_norm = float("nan")
            shape_distance_mean_norm = float("nan")
            shape_distance_max_norm = float("nan")
            r_shape = 0.0
        else:
            boundary_poly = _as_polyline(true_boundary_poly, "true_boundary_poly")
            distances = target_point_distances_to_polyline(reference_points, boundary_poly)
            radius_scale_safe = max(float(radius_scale), 1.0)
            shape_error_norm = float(np.sqrt(np.mean(distances**2)) / radius_scale_safe)
            shape_distance_mean_norm = float(np.mean(distances) / radius_scale_safe)
            shape_distance_max_norm = float(np.max(distances) / radius_scale_safe)
            r_shape = float(np.exp(-((shape_error_norm / max(float(self.shape_tolerance_norm), 1e-12)) ** 2)))
        action = np.asarray(action_norm, dtype=float).reshape(-1)
        prev = np.asarray(previous_action_norm, dtype=float).reshape(-1)
        if action.shape != prev.shape:
            raise ValueError("action_norm and previous_action_norm must have the same shape")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(prev)):
            raise ValueError("action vectors must contain finite values")
        action_rms = float(np.sqrt(np.mean(action**2))) if action.size else 0.0
        delta_action_rms = float(np.sqrt(np.mean((action - prev) ** 2))) if action.size else 0.0
        current_limit_penalty = _margin_penalty(current_limit_margin, weight=float(self.current_limit_weight))
        derivative_limit_penalty = _margin_penalty(derivative_limit_margin, weight=float(self.derivative_limit_weight))
        reward = (
            float(self.ip_weight) * r_ip
            + float(self.shape_weight) * r_shape
            - float(self.action_weight) * action_rms**2
            - float(self.delta_action_weight) * delta_action_rms**2
            - current_limit_penalty
            - derivative_limit_penalty
        )
        if terminated:
            reward -= float(self.termination_penalty)
        return reward, {
            "ip_error_norm": ip_error_norm,
            "shape_error_norm": shape_error_norm,
            "shape_distance_mean_norm": shape_distance_mean_norm,
            "shape_distance_max_norm": shape_distance_max_norm,
            "shape_target_point_count": float(reference_points.shape[0]),
            "r_ip": r_ip,
            "r_shape": r_shape,
            "action_rms": action_rms,
            "delta_action_rms": delta_action_rms,
            "current_limit_penalty": current_limit_penalty,
            "derivative_limit_penalty": derivative_limit_penalty,
        }


def boundary_points_from_radii(*, center: tuple[float, float], angles_rad: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Convert angle-sampled reference radii into Cartesian target shape points."""
    angles = np.asarray(angles_rad, dtype=float).reshape(-1)
    radius_values = np.asarray(radii, dtype=float).reshape(-1)
    if angles.shape != radius_values.shape:
        raise ValueError("angles_rad and radii must have the same shape")
    if angles.size == 0:
        raise ValueError("reference boundary must contain at least one target point")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(radius_values)):
        raise ValueError("reference boundary angles and radii must be finite")
    center_arr = np.asarray(center, dtype=float).reshape(-1)
    if center_arr.shape != (2,) or not np.all(np.isfinite(center_arr)):
        raise ValueError("center must contain two finite coordinates")
    return np.column_stack(
        [
            center_arr[0] + radius_values * np.cos(angles),
            center_arr[1] + radius_values * np.sin(angles),
        ]
    )


def target_point_distances_to_polyline(target_points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Measure the shortest distance from each target shape point to a boundary polyline."""
    points = _as_points(target_points, "target_points")
    poly = _as_polyline(polyline, "polyline")
    closed = _closed_polyline(poly)
    starts = closed[:-1]
    ends = closed[1:]
    seg = ends - starts
    seg_len2 = np.sum(seg**2, axis=1)
    keep = seg_len2 > 0.0
    if not np.any(keep):
        raise ValueError("polyline must contain at least one non-degenerate segment")
    starts = starts[keep]
    seg = seg[keep]
    seg_len2 = seg_len2[keep]
    diff = points[:, None, :] - starts[None, :, :]
    frac = np.sum(diff * seg[None, :, :], axis=2) / seg_len2[None, :]
    frac = np.clip(frac, 0.0, 1.0)
    closest = starts[None, :, :] + frac[:, :, None] * seg[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return np.min(distances, axis=1)


def _as_points(values: np.ndarray, name: str) -> np.ndarray:
    """Validate a finite non-empty point array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr.copy()


def _as_polyline(values: np.ndarray, name: str) -> np.ndarray:
    """Validate a finite boundary polyline."""
    arr = _as_points(values, name)
    if arr.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two points")
    return arr


def _closed_polyline(polyline: np.ndarray) -> np.ndarray:
    """Return a closed copy of a boundary polyline."""
    poly = np.asarray(polyline, dtype=float)
    if np.allclose(poly[0], poly[-1]):
        return poly.copy()
    return np.vstack([poly, poly[0]])


def _margin_penalty(margin: np.ndarray | None, *, weight: float) -> float:
    """Compute a quadratic penalty for low normalized safety margins."""
    if margin is None or weight == 0.0:
        return 0.0
    values = np.asarray(margin, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(weight) * float(np.mean(np.maximum(0.0, 0.10 - finite) ** 2))
