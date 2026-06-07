from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TerminationConfig:
    """Configurable environment-level episode termination rules."""

    terminate_on_boundary_loss: bool = True
    boundary_loss_grace_steps: int = 0
    terminate_on_nonfinite_observation: bool = True
    terminate_on_nonfinite_reward: bool = True
    current_limit_margin_min: float | None = None
    derivative_limit_margin_min: float | None = None
    measured_boundary_missing_steps: int | None = None

    def __post_init__(self) -> None:
        for name in ("current_limit_margin_min", "derivative_limit_margin_min"):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be >= 0 when provided")
        if self.measured_boundary_missing_steps is not None and int(self.measured_boundary_missing_steps) <= 0:
            raise ValueError("measured_boundary_missing_steps must be > 0 when provided")
        if int(self.boundary_loss_grace_steps) < 0:
            raise ValueError("boundary_loss_grace_steps must be >= 0")


@dataclass(frozen=True, slots=True)
class ReplayInitialStateCandidate:
    """One replay-derived initial state candidate for episode reset sampling."""

    shot: str
    initial_currents_path: Path
    initial_ip: float
    initial_boundary_parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.shot).strip():
            raise ValueError("replay initial-state shot must be non-empty")
        if not _is_finite(float(self.initial_ip)):
            raise ValueError("replay initial-state initial_ip must be finite")


@dataclass(frozen=True, slots=True)
class ReplayInitialStateConfig:
    """Replay-derived initial state sampling pool."""

    candidates: tuple[ReplayInitialStateCandidate, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("replay initial-state config requires at least one candidate")


@dataclass(frozen=True, slots=True)
class RangeInitialStateConfig:
    """Generated initial-state sampling bounds derived from replay starts."""

    ip: tuple[float, float]
    pfc_currents: tuple[tuple[float, float], ...]
    sol_currents: tuple[tuple[float, float], ...]
    boundary_parameters: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_interval(self.ip, "range initial-state ip")
        for bank, intervals in (("pfc_currents", self.pfc_currents), ("sol_currents", self.sol_currents)):
            if not intervals:
                raise ValueError(f"range initial-state {bank} requires at least one channel")
            for index, interval in enumerate(intervals):
                _validate_interval(interval, f"range initial-state {bank}[{index}]")
        required = {"R0", "Z0", "A0", "kappa", "delta"}
        missing = required.difference(self.boundary_parameters)
        if missing:
            raise ValueError(f"range initial-state boundary_parameters missing: {sorted(missing)}")
        for name, interval in self.boundary_parameters.items():
            _validate_interval(interval, f"range initial-state boundary_parameters.{name}")


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Configuration needed to construct a tokamak-sim backed RL environment."""

    sim_config_path: Path
    initial_currents_path: Path | None = None
    initial_ip: float | None = None
    initial_coil_currents: str = "config"
    initial_ip_scale: float | None = None
    replay_initial_state: ReplayInitialStateConfig | None = None
    range_initial_state: RangeInitialStateConfig | None = None
    scenario_name: str = "nominal"
    scenario_args: dict[str, object] = field(default_factory=dict)
    angles: int = 32
    max_episode_steps: int = 1000
    realism_enabled: bool = True
    resample_references_on_reset: bool = True
    observation_version: str = "v1"
    target_preview_steps: int = 0
    target_preview_stride: int = 1
    termination: TerminationConfig = field(default_factory=TerminationConfig)

    def __post_init__(self) -> None:
        version = str(self.observation_version)
        if version not in {"v1", "v2"}:
            raise ValueError("observation_version must be 'v1' or 'v2'")
        if int(self.target_preview_steps) < 0:
            raise ValueError("target_preview_steps must be >= 0")
        if int(self.target_preview_stride) <= 0:
            raise ValueError("target_preview_stride must be > 0")
        if version == "v1" and int(self.target_preview_steps) != 0:
            raise ValueError("target_preview_steps must be 0 for observation_version 'v1'")
        if version == "v2" and int(self.target_preview_steps) <= 0:
            raise ValueError("target_preview_steps must be > 0 for observation_version 'v2'")
        if self.initial_ip is not None and not _is_finite(float(self.initial_ip)):
            raise ValueError("initial_ip must be finite when provided")
        if str(self.initial_coil_currents) not in {"config", "zero", "sample_replay", "sample_ranges"}:
            raise ValueError("initial_coil_currents must be 'config', 'zero', 'sample_replay', or 'sample_ranges'")
        if str(self.initial_coil_currents) == "sample_replay" and self.replay_initial_state is None:
            raise ValueError("sample_replay initial_coil_currents requires replay_initial_state")
        if str(self.initial_coil_currents) == "sample_ranges" and self.range_initial_state is None:
            raise ValueError("sample_ranges initial_coil_currents requires range_initial_state")
        if self.initial_ip_scale is not None:
            scale = float(self.initial_ip_scale)
            if not _is_finite(scale) or scale <= 0.0:
                raise ValueError("initial_ip_scale must be finite and > 0 when provided")


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _validate_interval(interval: tuple[float, float], name: str) -> None:
    if len(interval) != 2:
        raise ValueError(f"{name} must contain two values")
    low = float(interval[0])
    high = float(interval[1])
    if not (_is_finite(low) and _is_finite(high)):
        raise ValueError(f"{name} bounds must be finite")
    if low > high:
        raise ValueError(f"{name} min must be <= max")
