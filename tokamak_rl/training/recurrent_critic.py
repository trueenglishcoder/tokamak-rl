from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from tokamak_rl.networks import FeedForwardActor, RecurrentQCritic
from tokamak_rl.training.sequence_replay import SequenceBatch


@dataclass(frozen=True, slots=True)
class RecurrentUpdateResult:
    critic_loss: float
    actor_loss: float | None
    valid_steps: int
    mpo_temperature_loss: float | None = None
    mpo_temperature: float | None = None
    mpo_mean_kl: float | None = None
    mpo_std_kl: float | None = None
    mpo_kl_dual_loss: float | None = None
    mpo_mean_kl_penalty: float | None = None
    mpo_std_kl_penalty: float | None = None


def recurrent_critic_update_once(
    batch: SequenceBatch,
    *,
    actor: FeedForwardActor,
    actor_target: FeedForwardActor,
    q1: RecurrentQCritic,
    q2: RecurrentQCritic,
    q1_target: RecurrentQCritic,
    q2_target: RecurrentQCritic,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    mpo_kl_opt: torch.optim.Optimizer | None = None,
    log_mpo_mean_kl_penalty: torch.Tensor | None = None,
    log_mpo_std_kl_penalty: torch.Tensor | None = None,
    gamma: float,
    tau: float,
    update_index: int,
    policy_delay: int,
    mpo_epsilon: float = 0.1,
    mpo_mean_kl_epsilon: float = 0.01,
    mpo_std_kl_epsilon: float = 1.0e-4,
    mpo_action_samples: int = 20,
    mpo_temperature_iterations: int = 10,
    mpo_temperature_lr: float = 0.1,
    mpo_initial_temperature: float = 1.0,
    torch_generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> RecurrentUpdateResult:
    """Run one asymmetric recurrent-critic MPO update over sampled sequence chunks."""
    if float(gamma) < 0.0 or float(gamma) > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if float(tau) <= 0.0 or float(tau) > 1.0:
        raise ValueError("tau must be in (0, 1]")
    if int(policy_delay) <= 0:
        raise ValueError("policy_delay must be > 0")
    if int(mpo_action_samples) <= 0:
        raise ValueError("mpo_action_samples must be > 0")
    if int(mpo_temperature_iterations) <= 0:
        raise ValueError("mpo_temperature_iterations must be > 0")
    for name, value in (
        ("mpo_epsilon", mpo_epsilon),
        ("mpo_mean_kl_epsilon", mpo_mean_kl_epsilon),
        ("mpo_std_kl_epsilon", mpo_std_kl_epsilon),
        ("mpo_temperature_lr", mpo_temperature_lr),
        ("mpo_initial_temperature", mpo_initial_temperature),
    ):
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be > 0")
    if mpo_kl_opt is not None and (log_mpo_mean_kl_penalty is None or log_mpo_std_kl_penalty is None):
        raise ValueError("MPO KL penalty tensors are required when mpo_kl_opt is set")
    device = torch.device(device)
    obs = _batch_tensor(batch.observations, dtype=torch.float32, device=device)
    actions = _batch_tensor(batch.actions, dtype=torch.float32, device=device)
    rewards = _batch_tensor(batch.rewards, dtype=torch.float32, device=device)
    next_obs = _batch_tensor(batch.next_observations, dtype=torch.float32, device=device)
    terminated = _batch_tensor(batch.terminated, dtype=torch.float32, device=device)
    mask = _batch_tensor(batch.mask, dtype=torch.float32, device=device)
    valid_steps = int(torch.sum(mask).detach().cpu().item())
    if valid_steps <= 0:
        raise ValueError("sequence batch mask has no valid timesteps")
    with torch.no_grad():
        next_action_samples, _next_log_prob = _actor_sample_many_for_sequence(
            actor_target,
            next_obs,
            sample_count=int(mpo_action_samples),
            torch_generator=torch_generator if device.type == "cpu" else None,
        )
        target_q_samples = _minimum_recurrent_q_samples(q1_target, q2_target, next_obs, next_action_samples)
        target_q = torch.mean(target_q_samples, dim=-1)
        target = rewards + float(gamma) * (1.0 - terminated) * target_q
    q1_pred = q1(obs, actions)
    q2_pred = q2(obs, actions)
    critic_loss = _masked_mse(q1_pred, target, mask) + _masked_mse(q2_pred, target, mask)
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_opt.step()

    actor_loss_value: float | None = None
    temperature_loss_value: float | None = None
    temperature_value: float | None = None
    mean_kl_value: float | None = None
    std_kl_value: float | None = None
    kl_dual_loss_value: float | None = None
    mean_kl_penalty_value: float | None = None
    std_kl_penalty_value: float | None = None
    if int(update_index) % int(policy_delay) == 0:
        with torch.no_grad():
            old_mean, old_std = _actor_distribution_for_sequence(actor, obs)
            actor_action_samples, _sample_log_prob = _actor_sample_many_for_sequence(
                actor,
                obs,
                sample_count=int(mpo_action_samples),
                torch_generator=torch_generator if device.type == "cpu" else None,
            )
            action_q = _minimum_recurrent_q_samples(q1, q2, obs, actor_action_samples)
            temperature, temperature_loss = _solve_mpo_temperature(
                action_q,
                mask,
                epsilon=float(mpo_epsilon),
                initial_temperature=float(mpo_initial_temperature),
                iterations=int(mpo_temperature_iterations),
                lr=float(mpo_temperature_lr),
            )
            action_weights = torch.softmax(action_q / torch.clamp(temperature, min=1.0e-6), dim=-1).detach()
        log_prob = _actor_log_prob_for_sequence_samples(actor, obs, actor_action_samples.detach())
        weighted_nll = -torch.sum(action_weights * log_prob * mask.unsqueeze(-1)) / torch.clamp(torch.sum(mask), min=1.0)
        new_mean, new_std = _actor_distribution_for_sequence(actor, obs)
        mean_kl, std_kl = _gaussian_kl_parts(old_mean, old_std, new_mean, new_std, mask)
        mean_penalty = _positive_penalty(log_mpo_mean_kl_penalty, device=device)
        std_penalty = _positive_penalty(log_mpo_std_kl_penalty, device=device)
        actor_loss = weighted_nll + mean_penalty.detach() * mean_kl + std_penalty.detach() * std_kl
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_opt.step()
        actor_loss_value = float(actor_loss.detach().cpu().item())
        temperature_loss_value = float(temperature_loss.detach().cpu().item())
        temperature_value = float(temperature.detach().cpu().item())
        mean_kl_value = float(mean_kl.detach().cpu().item())
        std_kl_value = float(std_kl.detach().cpu().item())
        mean_kl_penalty_value = float(mean_penalty.detach().cpu().item())
        std_kl_penalty_value = float(std_penalty.detach().cpu().item())
        if mpo_kl_opt is not None and log_mpo_mean_kl_penalty is not None and log_mpo_std_kl_penalty is not None:
            kl_dual_loss = -(
                log_mpo_mean_kl_penalty * (mean_kl.detach() - float(mpo_mean_kl_epsilon))
                + log_mpo_std_kl_penalty * (std_kl.detach() - float(mpo_std_kl_epsilon))
            )
            mpo_kl_opt.zero_grad(set_to_none=True)
            kl_dual_loss.backward()
            mpo_kl_opt.step()
            kl_dual_loss_value = float(kl_dual_loss.detach().cpu().item())

    _soft_update(q1_target, q1, tau=float(tau))
    _soft_update(q2_target, q2, tau=float(tau))
    _soft_update(actor_target, actor, tau=float(tau))
    critic_loss_value = float(critic_loss.detach().cpu().item())
    finite_values = [
        critic_loss_value,
        *(value for value in (actor_loss_value, temperature_loss_value, temperature_value, mean_kl_value, std_kl_value, kl_dual_loss_value, mean_kl_penalty_value, std_kl_penalty_value) if value is not None),
    ]
    if not all(np.isfinite(value) for value in finite_values):
        raise RuntimeError("recurrent update produced non-finite loss")
    return RecurrentUpdateResult(
        critic_loss=critic_loss_value,
        actor_loss=actor_loss_value,
        valid_steps=valid_steps,
        mpo_temperature_loss=temperature_loss_value,
        mpo_temperature=temperature_value,
        mpo_mean_kl=mean_kl_value,
        mpo_std_kl=std_kl_value,
        mpo_kl_dual_loss=kl_dual_loss_value,
        mpo_mean_kl_penalty=mean_kl_penalty_value,
        mpo_std_kl_penalty=std_kl_penalty_value,
    )


def _batch_tensor(value, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _actor_actions_for_sequence(actor: FeedForwardActor, observations: torch.Tensor) -> torch.Tensor:
    batch, time, obs_dim = observations.shape
    flat = observations.reshape(batch * time, obs_dim)
    actions = actor.deterministic_action(flat)
    return actions.reshape(batch, time, actor.action_dim)


def _actor_sample_for_sequence(actor: FeedForwardActor, observations: torch.Tensor, *, torch_generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    batch, time, obs_dim = observations.shape
    flat = observations.reshape(batch * time, obs_dim)
    actions, log_prob, _mean, _std = actor.sample_action_with_log_prob(flat, generator=torch_generator)
    return actions.reshape(batch, time, actor.action_dim), log_prob.reshape(batch, time)


def _actor_distribution_for_sequence(actor: FeedForwardActor, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, time, obs_dim = observations.shape
    flat = observations.reshape(batch * time, obs_dim)
    mean, std = actor.forward(flat)
    return mean.reshape(batch, time, actor.action_dim), std.reshape(batch, time, actor.action_dim)


def _actor_sample_many_for_sequence(
    actor: FeedForwardActor,
    observations: torch.Tensor,
    *,
    sample_count: int,
    torch_generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, time, obs_dim = observations.shape
    flat = observations.reshape(batch * time, obs_dim)
    mean, std = actor.forward(flat)
    noise = torch.randn(
        (flat.shape[0], int(sample_count), actor.action_dim),
        dtype=mean.dtype,
        device=mean.device,
        generator=torch_generator,
    )
    pre_tanh = mean[:, None, :] + std[:, None, :] * noise
    actions = torch.tanh(pre_tanh)
    log_prob = _squashed_gaussian_log_prob(mean[:, None, :], std[:, None, :], actions)
    return actions.reshape(batch, time, int(sample_count), actor.action_dim), log_prob.reshape(batch, time, int(sample_count))


def _actor_log_prob_for_sequence_samples(actor: FeedForwardActor, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    batch, time, sample_count, action_dim = actions.shape
    _ = sample_count
    if int(action_dim) != actor.action_dim:
        raise ValueError("sampled action dimension does not match actor action_dim")
    mean, std = _actor_distribution_for_sequence(actor, observations)
    return _squashed_gaussian_log_prob(mean[:, :, None, :], std[:, :, None, :], actions)


def _squashed_gaussian_log_prob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(action, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    pre_tanh = torch.atanh(clipped)
    normal_log_prob = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2.0 * torch.log(std) + math.log(2.0 * math.pi))
    squash_correction = torch.log(torch.clamp(1.0 - clipped.pow(2), min=1.0e-6))
    return torch.sum(normal_log_prob - squash_correction, dim=-1)


def _minimum_recurrent_q_samples(q1: RecurrentQCritic, q2: RecurrentQCritic, observations: torch.Tensor, action_samples: torch.Tensor) -> torch.Tensor:
    sample_count = int(action_samples.shape[2])
    values = []
    for sample_index in range(sample_count):
        sample_actions = action_samples[:, :, sample_index, :]
        values.append(torch.minimum(q1(observations, sample_actions), q2(observations, sample_actions)))
    return torch.stack(values, dim=-1)


def _solve_mpo_temperature(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
    initial_temperature: float,
    iterations: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        detached_q = q_values.detach()
        device = detached_q.device
        log_temperature = torch.tensor(math.log(float(initial_temperature)), dtype=detached_q.dtype, device=device, requires_grad=True)
        opt = torch.optim.Adam([log_temperature], lr=float(lr))
        loss = torch.as_tensor(float("nan"), dtype=detached_q.dtype, device=device)
        for _ in range(int(iterations)):
            temperature = F.softplus(log_temperature) + 1.0e-6
            log_partition = torch.logsumexp(detached_q / temperature, dim=-1) - math.log(int(detached_q.shape[-1]))
            objective = temperature * (float(epsilon) + log_partition)
            loss = torch.sum(objective * mask) / torch.clamp(torch.sum(mask), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        temperature = F.softplus(log_temperature.detach()) + 1.0e-6
        return temperature, loss.detach()


def _gaussian_kl_parts(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    old_var = torch.clamp(old_std.pow(2), min=1.0e-12)
    new_var = torch.clamp(new_std.pow(2), min=1.0e-12)
    mean_kl_per_dim = (new_mean - old_mean).pow(2) / (2.0 * old_var)
    std_ratio = new_var / old_var
    std_kl_per_dim = 0.5 * (std_ratio - 1.0 - torch.log(torch.clamp(std_ratio, min=1.0e-12)))
    denom = torch.clamp(torch.sum(mask), min=1.0)
    mean_kl = torch.sum(torch.sum(mean_kl_per_dim, dim=-1) * mask) / denom
    std_kl = torch.sum(torch.sum(std_kl_per_dim, dim=-1) * mask) / denom
    return mean_kl, std_kl


def _positive_penalty(log_value: torch.Tensor | None, *, device: torch.device) -> torch.Tensor:
    if log_value is None:
        return torch.as_tensor(1.0, dtype=torch.float32, device=device)
    return F.softplus(log_value) + 1.0e-6


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target) ** 2
    return torch.sum(err * mask) / torch.clamp(torch.sum(mask), min=1.0)


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, *, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.mul_(1.0 - float(tau)).add_(source_param, alpha=float(tau))


__all__ = ["RecurrentUpdateResult", "recurrent_critic_update_once"]
