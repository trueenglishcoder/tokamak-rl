from __future__ import annotations

from dataclasses import asdict, dataclass, field

from tokamak_control.realism import ActuatorRealismSettings, RealismSettings, SensorRealismSettings


@dataclass(frozen=True, slots=True)
class RandomizationSample:
    """One episode's sampled simulator randomization contract."""

    metadata: dict[str, object]
    realism_settings: RealismSettings | None


@dataclass(frozen=True, slots=True)
class DomainRandomizer:
    """Episode-level simulator noise/randomization settings.

    This intentionally exposes only perturbations already supported by
    tokamak-sim's neutral runtime realism layer. Plant-parameter randomization
    remains absent until the simulator exposes explicit plant hooks.
    """

    enabled: bool = False
    actuators: ActuatorRealismSettings = field(default_factory=ActuatorRealismSettings)
    sensors: SensorRealismSettings = field(default_factory=SensorRealismSettings)
    note: str | None = None

    def sample_episode(self, seed: int | None = None) -> RandomizationSample:
        settings = RealismSettings(
            enabled=bool(self.enabled),
            seed=None if seed is None else int(seed),
            actuators=self.actuators,
            sensors=self.sensors,
        )
        settings.validate()
        metadata = {
            "enabled": bool(self.enabled),
            "seed": seed,
            "has_nonzero_effect": _settings_have_nonzero_effect(settings),
            "simulator_realism": _realism_settings_dict(settings),
        }
        if self.note:
            metadata["note"] = str(self.note)
        return RandomizationSample(metadata=metadata, realism_settings=settings if bool(self.enabled) else None)


def _realism_settings_dict(settings: RealismSettings) -> dict[str, object]:
    data = asdict(settings)
    return {
        "enabled": bool(data["enabled"]),
        "seed": data["seed"],
        "actuators": dict(data["actuators"]),
        "sensors": dict(data["sensors"]),
    }


def _settings_have_nonzero_effect(settings: RealismSettings) -> bool:
    a = settings.actuators
    s = settings.sensors
    return any(
        (
            int(a.pfc_delay_steps) > 0,
            int(a.sol_delay_steps) > 0,
            float(a.pfc_gain_sigma) > 0.0,
            float(a.sol_gain_sigma) > 0.0,
            float(a.pfc_bias_sigma) > 0.0,
            float(a.sol_bias_sigma) > 0.0,
            float(a.pfc_command_noise_sigma) > 0.0,
            float(a.sol_command_noise_sigma) > 0.0,
            float(s.ip_noise_sigma) > 0.0,
            float(s.ip_bias) != 0.0,
            float(s.ip_bias_sigma) > 0.0,
            int(s.ip_delay_steps) > 0,
            float(s.active_current_noise_sigma) > 0.0,
            float(s.active_current_bias_sigma) > 0.0,
            int(s.active_current_delay_steps) > 0,
            float(s.radii_noise_sigma) > 0.0,
            float(s.radii_bias_sigma) > 0.0,
            int(s.radii_delay_steps) > 0,
            float(s.boundary_xy_noise_sigma) > 0.0,
            int(s.boundary_delay_steps) > 0,
            float(s.psi_noise_sigma) > 0.0,
        )
    )
