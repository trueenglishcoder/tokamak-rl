from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_simple_trainer import ToyContinuousEnv
from tokamak_rl.networks import ActorConfig, FeedForwardActor, RecurrentCriticConfig, RecurrentQCritic
from tokamak_rl.training import EpisodeReplayBuffer, recurrent_critic_update_once


def _collect_toy_episode(*, seed: int, max_steps: int = 5):
    env = ToyContinuousEnv(max_steps=max_steps)
    rng = np.random.default_rng(seed)
    obs, _info = env.reset(seed=seed)
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminated = []
    truncated = []
    try:
        for _ in range(max_steps):
            action = rng.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)
            next_obs, reward, term, trunc, _info = env.step(action)
            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            next_observations.append(np.asarray(next_obs, dtype=np.float32))
            terminated.append(bool(term))
            truncated.append(bool(trunc))
            obs = next_obs
            if term or trunc:
                break
    finally:
        env.close()
    return {
        "observations": np.stack(observations, axis=0),
        "actions": np.stack(actions, axis=0),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_observations": np.stack(next_observations, axis=0),
        "terminated": np.asarray(terminated, dtype=bool),
        "truncated": np.asarray(truncated, dtype=bool),
    }


def test_recurrent_q_critic_forward_shape_and_validation() -> None:
    critic = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=16, mlp_hidden_dim=16))
    obs = torch.randn(3, 5, 4)
    action = torch.randn(3, 5, 2)

    q = critic(obs, action)

    assert q.shape == (3, 5)
    assert torch.all(torch.isfinite(q))
    with pytest.raises(ValueError, match="observation shape"):
        critic(torch.randn(3, 4), action)
    with pytest.raises(ValueError, match="action shape"):
        critic(obs, torch.randn(3, 5, 3))


def test_episode_replay_samples_padded_sequence_masks() -> None:
    replay = EpisodeReplayBuffer(capacity_episodes=4, obs_dim=4, action_dim=2)
    replay.add_episode(**_collect_toy_episode(seed=1, max_steps=3))
    replay.add_episode(**_collect_toy_episode(seed=2, max_steps=5))

    batch = replay.sample_sequences(batch_size=6, sequence_length=6, rng=np.random.default_rng(4))

    assert batch.observations.shape == (6, 6, 4)
    assert batch.actions.shape == (6, 6, 2)
    assert batch.rewards.shape == (6, 6)
    assert batch.mask.shape == (6, 6)
    assert np.all(np.any(batch.mask, axis=1))
    assert np.any(~batch.mask)
    assert np.all(batch.rewards[~batch.mask] == 0.0)


def test_episode_replay_rejects_bad_episode_shapes() -> None:
    replay = EpisodeReplayBuffer(capacity_episodes=1, obs_dim=4, action_dim=2)
    episode = _collect_toy_episode(seed=3, max_steps=3)
    episode["actions"] = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="action array"):
        replay.add_episode(**episode)


def test_recurrent_critic_update_once_is_finite() -> None:
    torch.manual_seed(6)
    replay = EpisodeReplayBuffer(capacity_episodes=8, obs_dim=4, action_dim=2)
    for seed in range(6):
        replay.add_episode(**_collect_toy_episode(seed=10 + seed, max_steps=5))
    batch = replay.sample_sequences(batch_size=4, sequence_length=4, rng=np.random.default_rng(7))
    actor = FeedForwardActor(ActorConfig(obs_dim=4, action_dim=2, hidden_dim=16))
    actor_target = FeedForwardActor(ActorConfig(obs_dim=4, action_dim=2, hidden_dim=16))
    q1 = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=16, mlp_hidden_dim=16))
    q2 = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=16, mlp_hidden_dim=16))
    q1_target = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=16, mlp_hidden_dim=16))
    q2_target = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=16, mlp_hidden_dim=16))
    actor_target.load_state_dict(actor.state_dict())
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1.0e-3)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1.0e-3)
    log_mean_penalty = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    log_std_penalty = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    mpo_kl_opt = torch.optim.Adam([log_mean_penalty, log_std_penalty], lr=1.0e-3)

    result = recurrent_critic_update_once(
        batch,
        actor=actor,
        actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        actor_opt=actor_opt,
        critic_opt=critic_opt,
        log_mpo_mean_kl_penalty=log_mean_penalty,
        log_mpo_std_kl_penalty=log_std_penalty,
        mpo_kl_opt=mpo_kl_opt,
        mpo_action_samples=4,
        mpo_temperature_iterations=3,
        gamma=0.99,
        tau=0.01,
        update_index=2,
        policy_delay=2,
    )

    assert result.valid_steps == int(np.sum(batch.mask))
    assert np.isfinite(result.critic_loss)
    assert result.actor_loss is not None
    assert np.isfinite(result.actor_loss)
    assert result.mpo_temperature_loss is not None
    assert np.isfinite(result.mpo_temperature_loss)
    assert result.mpo_temperature is not None
    assert result.mpo_temperature > 0.0
    assert result.mpo_mean_kl is not None and np.isfinite(result.mpo_mean_kl)
    assert result.mpo_std_kl is not None and np.isfinite(result.mpo_std_kl)
    assert result.mpo_kl_dual_loss is not None and np.isfinite(result.mpo_kl_dual_loss)
    assert result.mpo_mean_kl_penalty is not None and result.mpo_mean_kl_penalty > 0.0
    assert result.mpo_std_kl_penalty is not None and result.mpo_std_kl_penalty > 0.0


def test_recurrent_update_rejects_empty_mask() -> None:
    replay = EpisodeReplayBuffer(capacity_episodes=1, obs_dim=4, action_dim=2)
    replay.add_episode(**_collect_toy_episode(seed=20, max_steps=2))
    batch = replay.sample_sequences(batch_size=1, sequence_length=2, rng=np.random.default_rng(1))
    batch.mask[:] = False
    actor = FeedForwardActor(ActorConfig(obs_dim=4, action_dim=2, hidden_dim=8))
    q1 = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=8, mlp_hidden_dim=8))
    q2 = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=8, mlp_hidden_dim=8))
    q1_target = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=8, mlp_hidden_dim=8))
    q2_target = RecurrentQCritic(RecurrentCriticConfig(obs_dim=4, action_dim=2, hidden_dim=8, mlp_hidden_dim=8))
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1.0e-3)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1.0e-3)

    with pytest.raises(ValueError, match="no valid"):
        recurrent_critic_update_once(
            batch,
            actor=actor,
            actor_target=actor,
            q1=q1,
            q2=q2,
            q1_target=q1_target,
            q2_target=q2_target,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            gamma=0.99,
            tau=0.01,
            update_index=1,
            policy_delay=1,
        )
