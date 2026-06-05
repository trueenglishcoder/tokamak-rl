from __future__ import annotations

import numpy as np

from tokamak_rl.observations.schema import ObservationSchema


def build_observation(
    *,
    schema: ObservationSchema,
    step_index: int,
    max_episode_steps: int,
    measured_ip: float,
    ip_ref: float,
    ip_scale: float,
    measured_active_currents: np.ndarray,
    current_scale: np.ndarray,
    measured_radii: np.ndarray | None,
    radii_ref: np.ndarray,
    target_preview_time_norm: np.ndarray | None = None,
    ip_ref_preview: np.ndarray | None = None,
    radii_ref_preview: np.ndarray | None = None,
    radius_scale: float,
    previous_action_norm: np.ndarray,
) -> np.ndarray:
    """Build the first flat measured-channel actor observation."""
    n_active = int(schema.n_active_total)
    n_angles = int(schema.n_angles)
    currents = np.asarray(measured_active_currents, dtype=float).reshape(-1)
    current_scale_arr = np.asarray(current_scale, dtype=float).reshape(-1)
    ref = np.asarray(radii_ref, dtype=float).reshape(-1)
    prev = np.asarray(previous_action_norm, dtype=float).reshape(-1)
    _require_finite(np.array([measured_ip, ip_ref, ip_scale, radius_scale], dtype=float), "scalar observation inputs")
    if currents.shape != (n_active,) or current_scale_arr.shape != (n_active,) or prev.shape != (n_active,):
        raise ValueError("active-current and previous-action fields must match n_active_total")
    if ref.shape != (n_angles,):
        raise ValueError("radii_ref must match n_angles")
    _require_finite(currents, "measured_active_currents")
    _require_finite(current_scale_arr, "current_scale")
    _require_finite(ref, "radii_ref")
    _require_finite(prev, "previous_action_norm")
    if measured_radii is None:
        boundary_valid = 0.0
        radii = np.zeros((n_angles,), dtype=float)
        radii_error = np.zeros((n_angles,), dtype=float)
    else:
        boundary_valid = 1.0
        radii = np.asarray(measured_radii, dtype=float).reshape(-1)
        if radii.shape != (n_angles,):
            raise ValueError("measured_radii must match n_angles")
        _require_finite(radii, "measured_radii")
        radii_error = radii - ref
    phase = float(step_index) / max(float(max_episode_steps), 1.0)
    ip_scale_safe = max(float(ip_scale), 1.0)
    radius_scale_safe = max(float(radius_scale), 1.0)
    current_scale_safe = np.where(current_scale_arr > 0.0, current_scale_arr, 1.0)
    parts = [
        np.array([
            phase,
            boundary_valid,
            float(measured_ip) / ip_scale_safe,
            float(ip_ref) / ip_scale_safe,
            (float(measured_ip) - float(ip_ref)) / ip_scale_safe,
        ]),
        currents / current_scale_safe,
        radii / radius_scale_safe,
        ref / radius_scale_safe,
        radii_error / radius_scale_safe,
        np.clip(prev, -1.0, 1.0),
    ]
    if schema.version == "v2":
        preview_steps = int(schema.target_preview_steps)
        preview_t = _preview_vector(target_preview_time_norm, preview_steps, "target_preview_time_norm")
        preview_ip = _preview_vector(ip_ref_preview, preview_steps, "ip_ref_preview")
        preview_radii = _preview_matrix(radii_ref_preview, preview_steps, n_angles, "radii_ref_preview")
        parts.extend([
            preview_t,
            preview_ip / ip_scale_safe,
            preview_radii.reshape(-1) / radius_scale_safe,
        ])
    obs = np.concatenate(parts)
    if obs.shape != (schema.obs_dim,):
        raise ValueError(f"observation shape {obs.shape} != ({schema.obs_dim},)")
    _require_finite(obs, "observation")
    return obs.astype(np.float32, copy=False)


def _require_finite(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError(f"{name} must contain finite values")


def _preview_vector(values: np.ndarray | None, size: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required for observation schema v2")
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape != (int(size),):
        raise ValueError(f"{name} must have shape ({int(size)},)")
    _require_finite(arr, name)
    return arr


def _preview_matrix(values: np.ndarray | None, rows: int, cols: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required for observation schema v2")
    arr = np.asarray(values, dtype=float)
    if arr.shape != (int(rows), int(cols)):
        raise ValueError(f"{name} must have shape ({int(rows)}, {int(cols)})")
    _require_finite(arr, name)
    return arr
