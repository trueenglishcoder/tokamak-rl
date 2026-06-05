from __future__ import annotations

import pytest
import torch

from tokamak_rl.networks import ActorConfig, FeedForwardActor


def test_actor_forward_shapes_and_range() -> None:
    actor = FeedForwardActor(ActorConfig(obs_dim=18, action_dim=3))
    obs = torch.randn(5, 18)

    mean, std = actor(obs)
    action = actor.deterministic_action(obs)

    assert mean.shape == (5, 3)
    assert std.shape == (5, 3)
    assert action.shape == (5, 3)
    assert torch.all(torch.isfinite(mean))
    assert torch.all(torch.isfinite(std))
    assert torch.all(torch.isfinite(action))
    assert torch.all(std > 0.0)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_actor_deterministic_path_is_stable() -> None:
    torch.manual_seed(3)
    actor = FeedForwardActor(ActorConfig(obs_dim=18, action_dim=3))
    obs = torch.randn(2, 18)

    first = actor.deterministic_action(obs)
    second = actor.deterministic_action(obs)

    assert torch.allclose(first, second)


def test_actor_sample_path_is_seed_reproducible_and_bounded() -> None:
    actor = FeedForwardActor(ActorConfig(obs_dim=18, action_dim=3))
    obs = torch.randn(4, 18)
    gen_a = torch.Generator().manual_seed(9)
    gen_b = torch.Generator().manual_seed(9)

    action_a, mean_a, std_a = actor.sample_action(obs, generator=gen_a)
    action_b, mean_b, std_b = actor.sample_action(obs, generator=gen_b)

    assert torch.allclose(action_a, action_b)
    assert torch.allclose(mean_a, mean_b)
    assert torch.allclose(std_a, std_b)
    assert torch.all(action_a <= 1.0)
    assert torch.all(action_a >= -1.0)


def test_actor_sample_log_prob_path_is_seed_reproducible() -> None:
    actor = FeedForwardActor(ActorConfig(obs_dim=18, action_dim=3))
    obs = torch.randn(4, 18)
    gen_a = torch.Generator().manual_seed(11)
    gen_b = torch.Generator().manual_seed(11)

    action_a, log_prob_a, mean_a, std_a = actor.sample_action_with_log_prob(obs, generator=gen_a)
    action_b, log_prob_b, mean_b, std_b = actor.sample_action_with_log_prob(obs, generator=gen_b)

    assert action_a.shape == (4, 3)
    assert log_prob_a.shape == (4,)
    assert torch.allclose(action_a, action_b)
    assert torch.allclose(log_prob_a, log_prob_b)
    assert torch.allclose(mean_a, mean_b)
    assert torch.allclose(std_a, std_b)
    assert torch.all(torch.isfinite(log_prob_a))


def test_actor_rejects_bad_observation_shape_and_nonfinite_values() -> None:
    actor = FeedForwardActor(ActorConfig(obs_dim=18, action_dim=3))

    with pytest.raises(ValueError, match="observation shape"):
        actor.deterministic_action(torch.zeros(18))
    with pytest.raises(ValueError, match="observation shape"):
        actor.deterministic_action(torch.zeros(2, 19))
    with pytest.raises(ValueError, match="finite"):
        bad = torch.zeros(2, 18)
        bad[0, 0] = torch.nan
        actor.deterministic_action(bad)


def test_actor_config_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="obs_dim"):
        ActorConfig(obs_dim=0, action_dim=3)
    with pytest.raises(ValueError, match="action_dim"):
        ActorConfig(obs_dim=18, action_dim=0)
