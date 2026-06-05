from __future__ import annotations

import numpy as np
import pytest

from tokamak_rl.actions import ActionScaler
from tokamak_rl.observations import ObservationSchema
from tokamak_rl.observations.builder import build_observation
from tokamak_rl.randomization import DomainRandomizer
from tokamak_rl.rewards import JointCurrentBoundaryReward, target_point_distances_to_polyline


def _diamond_boundary() -> np.ndarray:
    """Return a simple closed boundary polyline for reward tests."""
    return np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]], dtype=float)


def test_action_scaler_maps_normalized_to_physical_derivatives() -> None:
    scaler = ActionScaler(np.array([10.0, 20.0], dtype=float))

    physical = scaler.to_physical(np.array([-1.0, 0.5], dtype=float))

    assert scaler.action_dim == 2
    assert np.allclose(physical, np.array([-10.0, 10.0]))
    assert np.allclose(scaler.to_normalized(physical), np.array([-1.0, 0.5]))


def test_action_scaler_clips_actions_and_rejects_bad_inputs() -> None:
    scaler = ActionScaler(np.array([10.0, 20.0], dtype=float))

    assert np.allclose(scaler.to_physical(np.array([2.0, -3.0])), np.array([10.0, -20.0]))
    with pytest.raises(ValueError, match="action shape"):
        scaler.to_physical(np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        scaler.to_physical(np.array([np.nan, 0.0]))


def test_action_scaler_requires_explicit_fallback_for_invalid_scale() -> None:
    with pytest.raises(ValueError, match="explicit fallback"):
        ActionScaler(np.array([np.nan, 10.0]))

    scaler = ActionScaler(np.array([np.nan, 10.0]), fallback_derivative_scale=5.0)

    assert np.allclose(scaler.derivative_scale, np.array([5.0, 10.0]))


def test_observation_schema_dimension_and_builder() -> None:
    schema = ObservationSchema(n_active_total=2, n_angles=3)

    obs = build_observation(
        schema=schema,
        step_index=5,
        max_episode_steps=10,
        measured_ip=90.0,
        ip_ref=100.0,
        ip_scale=100.0,
        measured_active_currents=np.array([1.0, -2.0]),
        current_scale=np.array([10.0, 10.0]),
        measured_radii=np.array([1.0, 1.1, 1.2]),
        radii_ref=np.array([1.0, 1.0, 1.0]),
        radius_scale=2.0,
        previous_action_norm=np.array([0.25, -0.5]),
    )

    assert schema.obs_dim == 18
    assert obs.shape == (18,)
    assert obs.dtype == np.float32
    assert obs[0] == 0.5
    assert obs[1] == 1.0
    assert schema.field_order == (
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
    assert schema.field_slices["previous_action_norm"] == slice(16, 18)
    assert schema.to_metadata()["obs_dim"] == 18


def test_observation_schema_v2_appends_target_preview_fields() -> None:
    schema = ObservationSchema(n_active_total=2, n_angles=3, version="v2", target_preview_steps=2)

    obs = build_observation(
        schema=schema,
        step_index=5,
        max_episode_steps=10,
        measured_ip=90.0,
        ip_ref=100.0,
        ip_scale=100.0,
        measured_active_currents=np.array([1.0, -2.0]),
        current_scale=np.array([10.0, 10.0]),
        measured_radii=np.array([1.0, 1.1, 1.2]),
        radii_ref=np.array([1.0, 1.0, 1.0]),
        target_preview_time_norm=np.array([0.1, 0.2]),
        ip_ref_preview=np.array([110.0, 120.0]),
        radii_ref_preview=np.array([[1.0, 1.1, 1.2], [1.2, 1.3, 1.4]]),
        radius_scale=2.0,
        previous_action_norm=np.array([0.25, -0.5]),
    )

    assert schema.obs_dim == 28
    assert obs.shape == (28,)
    assert schema.field_slices["target_preview_time_norm"] == slice(18, 20)
    assert schema.field_slices["ip_ref_preview_norm"] == slice(20, 22)
    assert schema.field_slices["radii_ref_preview_norm"] == slice(22, 28)
    assert np.allclose(obs[schema.field_slices["target_preview_time_norm"]], np.array([0.1, 0.2], dtype=np.float32))
    assert np.allclose(obs[schema.field_slices["ip_ref_preview_norm"]], np.array([1.1, 1.2], dtype=np.float32))
    assert schema.to_metadata()["schema_version"] == "v2"
    assert schema.to_metadata()["target_preview_steps"] == 2


def test_missing_boundary_observation_sets_valid_flag_zero() -> None:
    schema = ObservationSchema(n_active_total=1, n_angles=2)

    obs = build_observation(
        schema=schema,
        step_index=0,
        max_episode_steps=10,
        measured_ip=1.0,
        ip_ref=1.0,
        ip_scale=1.0,
        measured_active_currents=np.array([0.0]),
        current_scale=np.array([1.0]),
        measured_radii=None,
        radii_ref=np.array([1.0, 1.0]),
        radius_scale=1.0,
        previous_action_norm=np.array([0.0]),
    )

    assert obs[1] == 0.0
    slices = schema.field_slices
    assert np.allclose(obs[slices["radii_meas_norm"]], np.zeros((2,)))
    assert np.allclose(obs[slices["radii_error_norm"]], np.zeros((2,)))


def test_observation_builder_rejects_wrong_shapes_and_nonfinite_values() -> None:
    schema = ObservationSchema(n_active_total=1, n_angles=2)
    kwargs = dict(
        schema=schema,
        step_index=0,
        max_episode_steps=10,
        measured_ip=1.0,
        ip_ref=1.0,
        ip_scale=1.0,
        measured_active_currents=np.array([0.0]),
        current_scale=np.array([1.0]),
        measured_radii=np.array([1.0, 1.0]),
        radii_ref=np.array([1.0, 1.0]),
        radius_scale=1.0,
        previous_action_norm=np.array([0.0]),
    )

    bad_shape = dict(kwargs)
    bad_shape["measured_radii"] = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="measured_radii"):
        build_observation(**bad_shape)

    bad_finite = dict(kwargs)
    bad_finite["measured_ip"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_observation(**bad_finite)


def test_target_point_distances_measure_nearest_boundary_segments() -> None:
    """Reward geometry uses target-point distances to boundary segments."""
    distances = target_point_distances_to_polyline(
        np.array([[0.5, 0.5], [2.0, 0.0]], dtype=float),
        _diamond_boundary(),
    )

    assert np.allclose(distances, np.array([0.0, 1.0]))


def test_reward_prefers_tracking_and_penalizes_termination() -> None:
    reward_fn = JointCurrentBoundaryReward()
    good, _ = reward_fn(
        true_ip=100.0,
        ip_ref=100.0,
        ip_scale=100.0,
        true_boundary_poly=_diamond_boundary(),
        reference_boundary_points=_diamond_boundary()[:-1],
        radius_scale=1.0,
        action_norm=np.array([0.0]),
        previous_action_norm=np.array([0.0]),
        terminated=False,
    )
    bad, _ = reward_fn(
        true_ip=50.0,
        ip_ref=100.0,
        ip_scale=100.0,
        true_boundary_poly=None,
        reference_boundary_points=_diamond_boundary()[:-1],
        radius_scale=1.0,
        action_norm=np.array([1.0]),
        previous_action_norm=np.array([0.0]),
        terminated=True,
    )

    assert good > bad


def test_reward_margin_penalties_are_safe_and_inspectable() -> None:
    reward_fn = JointCurrentBoundaryReward(current_limit_weight=2.0, derivative_limit_weight=3.0)
    base_kwargs = dict(
        true_ip=100.0,
        ip_ref=100.0,
        ip_scale=100.0,
        true_boundary_poly=_diamond_boundary(),
        reference_boundary_points=_diamond_boundary()[:-1],
        radius_scale=1.0,
        action_norm=np.array([0.0]),
        previous_action_norm=np.array([0.0]),
        terminated=False,
    )

    no_margin_reward, no_margin_components = reward_fn(
        **base_kwargs,
        current_limit_margin=None,
        derivative_limit_margin=None,
    )
    low_margin_reward, low_margin_components = reward_fn(
        **base_kwargs,
        current_limit_margin=np.array([0.05, 0.20]),
        derivative_limit_margin=np.array([0.00, 0.30]),
    )

    assert no_margin_components["current_limit_penalty"] == 0.0
    assert no_margin_components["derivative_limit_penalty"] == 0.0
    assert low_margin_components["current_limit_penalty"] > 0.0
    assert low_margin_components["derivative_limit_penalty"] > 0.0
    assert low_margin_reward < no_margin_reward


def test_reward_rejects_mismatched_action_shapes() -> None:
    reward_fn = JointCurrentBoundaryReward()

    with pytest.raises(ValueError, match="same shape"):
        reward_fn(
            true_ip=1.0,
            ip_ref=1.0,
            ip_scale=1.0,
            true_boundary_poly=_diamond_boundary(),
            reference_boundary_points=_diamond_boundary()[:-1],
            radius_scale=1.0,
            action_norm=np.array([0.0, 0.0]),
            previous_action_norm=np.array([0.0]),
            terminated=False,
        )


def test_domain_randomizer_disabled_contract() -> None:
    sample = DomainRandomizer(enabled=False).sample_episode(seed=123)

    assert sample.metadata["enabled"] is False
    assert sample.metadata["seed"] == 123
    assert sample.metadata["has_nonzero_effect"] is False
    assert sample.realism_settings is None
