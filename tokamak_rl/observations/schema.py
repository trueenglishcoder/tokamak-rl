from __future__ import annotations

from dataclasses import dataclass


FIELD_ORDER = (
    "phase_norm",
    "boundary_valid",
    "ip_meas_norm",
    "ip_ref_norm",
    "ip_error_norm",
    "active_currents_meas_norm",
    "radii_meas_norm",
    "radii_ref_norm",
    "radii_error_norm",
    "previous_action_norm",
)

PREVIEW_FIELD_ORDER = (
    "target_preview_time_norm",
    "ip_ref_preview_norm",
    "radii_ref_preview_norm",
)


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    """Fixed flat observation schema for the first environment version."""

    n_active_total: int
    n_angles: int
    version: str = "v1"
    target_preview_steps: int = 0

    def __post_init__(self) -> None:
        if int(self.n_active_total) <= 0:
            raise ValueError("n_active_total must be > 0")
        if int(self.n_angles) <= 0:
            raise ValueError("n_angles must be > 0")
        if str(self.version) not in {"v1", "v2"}:
            raise ValueError("observation schema version must be 'v1' or 'v2'")
        if int(self.target_preview_steps) < 0:
            raise ValueError("target_preview_steps must be >= 0")
        if str(self.version) == "v1" and int(self.target_preview_steps) != 0:
            raise ValueError("v1 observations cannot include target preview fields")
        if str(self.version) == "v2" and int(self.target_preview_steps) <= 0:
            raise ValueError("v2 observations require target_preview_steps > 0")

    @property
    def field_order(self) -> tuple[str, ...]:
        if self.version == "v1":
            return FIELD_ORDER
        return FIELD_ORDER + PREVIEW_FIELD_ORDER

    @property
    def obs_dim(self) -> int:
        base = 5 + 2 * int(self.n_active_total) + 3 * int(self.n_angles)
        if self.version == "v1":
            return base
        return base + int(self.target_preview_steps) * (2 + int(self.n_angles))

    @property
    def field_sizes(self) -> dict[str, int]:
        sizes = {
            "phase_norm": 1,
            "boundary_valid": 1,
            "ip_meas_norm": 1,
            "ip_ref_norm": 1,
            "ip_error_norm": 1,
            "active_currents_meas_norm": int(self.n_active_total),
            "radii_meas_norm": int(self.n_angles),
            "radii_ref_norm": int(self.n_angles),
            "radii_error_norm": int(self.n_angles),
            "previous_action_norm": int(self.n_active_total),
        }
        if self.version == "v2":
            preview_steps = int(self.target_preview_steps)
            sizes.update(
                {
                    "target_preview_time_norm": preview_steps,
                    "ip_ref_preview_norm": preview_steps,
                    "radii_ref_preview_norm": preview_steps * int(self.n_angles),
                }
            )
        return sizes

    @property
    def field_slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        start = 0
        sizes = self.field_sizes
        for name in self.field_order:
            stop = start + sizes[name]
            out[name] = slice(start, stop)
            start = stop
        if start != self.obs_dim:
            raise RuntimeError("observation schema size accounting is inconsistent")
        return out

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.version,
            "obs_dim": self.obs_dim,
            "n_active_total": int(self.n_active_total),
            "n_angles": int(self.n_angles),
            "target_preview_steps": int(self.target_preview_steps),
            "field_order": list(self.field_order),
            "field_sizes": dict(self.field_sizes),
        }
