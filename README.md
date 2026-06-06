# Tokamak RL

Companion reinforcement-learning repository for `tokamak-sim`.

`tokamak-rl` contains the training-side code for tokamak-control experiments: environments, action scaling, observation construction, rewards, domain randomization, policy networks, critic networks, replay buffers, rollout evaluation, policy export, and training loops.

`tokamak-sim` remains the simulator/runtime repository. `tokamak-rl` imports `tokamak-sim` as a Python dependency and uses its public programmatic bridge. It does not run `tokamak-sim` as a separate simulator service.

## Quick Start

Expected local sibling layout:

```text
tokamak/
  tokamak-sim/
  tokamak-rl/
```

Local editable install from the sibling layout:

```bash
cd tokamak-rl
python -m pip install -e ../tokamak-sim
python -m pip install -e '.[train,dev]'
```

Run tests:

```bash
python -m pytest -q
```

Run a zero-policy evaluation from an experiment config:

```bash
python scripts/evaluate.py   --config configs/experiments/t15md_joint_current_boundary.yaml   --out outputs/evaluation   --episodes 1   --policy zero   --seed 0

Synthetic reference config:

```bash
python scripts/evaluate.py   --config configs/experiments/t15md_synthetic_joint_current_boundary.yaml   --out outputs/evaluation_synthetic   --episodes 1   --policy zero   --seed 0
```

Training-reference presets:

```text
configs/experiments/t15md_training_real_replay_like.yaml
configs/experiments/t15md_training_circle_static_boundary.yaml
```

The real-replay-like preset uses observed T15 Ip bounds plus fitted boundary parameter bounds/rate limits. The circular preset keeps the boundary fixed as a circle with `kappa = 1.0`, `delta = 0.0`, and constant `R0`, `Z0`, and `A0`, while still using continuous segmented Ip targets.

With `resample_references_on_reset: true`, the same evaluation seed is reproducible and different episode seeds produce different synthetic references. Rollout artifacts store `ip_ref`, `radii_ref`, and effective reference seeds for target diagnostics.
```

Run a random-policy evaluation:

```bash
python scripts/evaluate.py   --config configs/experiments/t15md_joint_current_boundary.yaml   --out outputs/evaluation_random   --episodes 1   --policy random   --seed 0
```

Run a short TCV-style recurrent-critic training smoke run:

```bash
python scripts/train.py   --config configs/experiments/t15md_training_real_replay_like_smoke.yaml
```

That smoke config uses the real T15 simulator config, real Ip template data, robust replay-derived boundary parameter bounds/rate limits, fixed validation seed, checkpoint cadence, and automatic `exports/best_actor/` export. Command-line flags such as `--steps`, `--device`, `--output-dir`, and `--checkpoint-dir` override the YAML values when supplied.

Run a short simple actor-critic training smoke run:

```bash
python scripts/train.py   --trainer simple   --config configs/experiments/t15md_joint_current_boundary.yaml   --steps 500   --warmup-steps 100   --batch-size 64   --hidden-dim 256   --num-envs 1   --output-dir outputs/train_simple   --checkpoint-dir checkpoints/train_simple   --eval-episodes 1   --eval-max-steps 200   --seed 0
```

Export a deterministic feedforward actor artifact:

```bash
python scripts/export_policy.py   --out outputs/exported_policy   --obs-dim 82   --action-dim 18   --n-active-total 18   --n-angles 32
```

Add `--checkpoint <path>` to export an existing PyTorch actor state dict or a full trainer checkpoint such as `checkpoints/.../best.pt`.

## Docker

`tokamak-rl` has its own Docker image for training. The image installs `tokamak-sim` as a Python dependency and the compose file mounts the sibling `tokamak-sim` source tree read-only at runtime. You do not launch the `tokamak-sim` Docker service for RL training.

Build from the `tokamak-rl` directory:

```bash
cd /home/mnsa/tokamak/tokamak-rl
docker compose build tokamak-rl
```

Run the 10-step static-circle smoke training on CPU:

```bash
docker compose run --rm tokamak-rl \
  python scripts/train.py \
    --config configs/experiments/t15md_training_circle_static_boundary.yaml \
    --trainer tcv_style \
    --steps 10 \
    --num-envs 2 \
    --warmup-steps 2 \
    --batch-size 1 \
    --sequence-length 2 \
    --hidden-dim 16 \
    --critic-hidden-dim 16 \
    --critic-mlp-hidden-dim 16 \
    --mpo-action-samples 4 \
    --mpo-temperature-iterations 3 \
    --updates-per-env-step 1 \
    --updates-per-episode 1 \
    --eval-episodes 2 \
    --eval-max-steps 10 \
    --eval-randomization-mode clean \
    --device cpu \
    --output-dir outputs/zero_start_circle_10x2 \
    --checkpoint-dir outputs/zero_start_circle_10x2/checkpoints \
    --checkpoint-interval-steps 5 \
    --max-step-checkpoints 2
```

On a server with the NVIDIA container runtime available, use the GPU-profile service and request CUDA:

```bash
docker compose --profile gpu run --rm tokamak-rl-gpu \
  python scripts/train.py \
    --config configs/experiments/t15md_training_real_replay_like.yaml \
    --device cuda
```

Training artifacts are written into the mounted `outputs/` or `checkpoints/` paths in this repository.

Weights & Biases logging is optional. Enable it from the CLI with `--wandb`:

```bash
python scripts/train.py \
  --config configs/experiments/t15md_training_circle_static_boundary.yaml \
  --trainer tcv_style \
  --wandb \
  --wandb-project tokamak-rl \
  --wandb-name circle-static-debug \
  --wandb-mode online
```

Use `--wandb-mode offline` for server runs without immediate upload, `--wandb-log-interval-steps N` to reduce scalar logging frequency, and `--wandb-no-artifacts` if you do not want metrics/checkpoints/export folders uploaded as W&B artifacts. W&B can also be configured in experiment YAML:

```yaml
wandb:
  enabled: true
  project: tokamak-rl
  name: circle-static-debug
  mode: online
  tags: circle, debug
  log_interval_steps: 10
  log_artifacts: true
```

Reward values can be searched with short controlled training runs before committing to final-length training. The search ranks candidates by deterministic evaluation diagnostics: mean normalized Ip error, mean normalized target-boundary point error, and boundary failure rate. Raw return is recorded but is not used for ranking because reward scale changes between candidates.

```bash
python scripts/search_rewards.py \
  --config configs/experiments/t15md_training_circle_static_boundary.yaml \
  --trainer tcv_style \
  --steps 200 \
  --num-envs 2 \
  --eval-episodes 2 \
  --eval-max-steps 100 \
  --device cuda \
  --output-dir outputs/reward_search_circle \
  --max-candidates 12 \
  --ip-weight-values 0.5,1.0,2.0 \
  --shape-weight-values 0.5,1.0,2.0 \
  --action-weight-values 0.001,0.01 \
  --delta-action-weight-values 0.001,0.01 \
  --termination-penalty-values 5.0,10.0,20.0
```

The search writes `results.csv`, `results.json`, `search_manifest.json`, per-candidate folders with `reward.yaml` and normal training artifacts, and `best_reward.yaml` for the best ranked candidate. Checkpoints are disabled by default during search; add `--save-checkpoints` only when you intentionally want candidate checkpoints.

## Current Layout

```text
README.md         Project overview and common commands
AGENTS.md         Coding rules for AI-assisted work
pyproject.toml    Package metadata and dependencies
configs/          Experiment, reward, and randomization presets
docs/             Project documentation
scripts/          Training, evaluation, rollout, and export entry points
tests/            Regression and smoke tests
tokamak_rl/       Importable RL package
```

Local ignored folders expected for normal work:

```text
runs/             Rollout, evaluation, and training artifacts
output/           Diagnostics, plots, exported summaries
outputs/          Script outputs when using default evaluation paths
checkpoints/      Training checkpoints
_local_archive/   Local notes and scratch files
```

## Main Commands

The source tree currently has these user-facing scripts:

```text
scripts/train.py
scripts/evaluate.py
scripts/rollout_policy.py
scripts/export_policy.py
```

The reviewed evaluation CLI supports:

```text
--config    experiment YAML path
--out       output directory
--episodes  number of evaluation episodes
--policy    zero or random
--seed      base random seed
```

`scripts/rollout_policy.py` is an equivalent rollout/evaluation entry point for zero and random policy artifact generation:

```bash
python scripts/rollout_policy.py --config configs/experiments/t15md_joint_current_boundary.yaml --out outputs/rollout_zero --episodes 1 --policy zero --seed 0
```

The reviewed training CLI supports:

```text
--trainer                 tcv_style or simple
--config                  experiment YAML path
--steps                   total environment steps
--warmup-steps            random-action warmup steps
--batch-size              replay update batch size
--sequence-length         recurrent sequence length for tcv_style
--hidden-dim              actor and default critic hidden width
--critic-hidden-dim       recurrent critic LSTM width
--critic-mlp-hidden-dim   recurrent critic MLP width
--mpo-kl-lr               optimizer learning rate for MPO KL dual variables
--mpo-epsilon             sampled-action E-step KL bound
--mpo-mean-kl-epsilon     actor mean KL bound
--mpo-std-kl-epsilon      actor standard-deviation KL bound
--mpo-action-samples      sampled actions per state for MPO policy improvement
--mpo-temperature-iterations optimizer iterations for the MPO E-step temperature
--mpo-temperature-lr      optimizer learning rate for the MPO E-step temperature
--device                  cpu, cuda, or auto learner device
--process-envs            run each training env in its own simulator worker process
--process-start-method    spawn, fork, or forkserver for --process-envs
--seed                    random seed
--checkpoint-dir          checkpoint output directory
--checkpoint-interval-steps optional numbered checkpoint/latest interval
--resume-checkpoint       checkpoint path for weight resume
--output-dir              metrics/losses/config output directory
--num-envs                synchronous training environment count
--updates-per-episode     tcv_style updates after each completed episode
--updates-per-env-step    tcv_style updates after each collected env step when replay is nonempty
--max-learner-catchup-updates optional per-trigger learner update cap
--eval-interval-steps     optional periodic deterministic evaluation interval
--eval-episodes           deterministic evaluation episode count
--eval-max-steps          maximum steps per evaluation episode
--no-progress             disable the terminal progress bar
```

The reviewed export CLI supports:

```text
--out             output directory for exported policy files
--obs-dim         actor observation dimension
--action-dim      actor action dimension
--n-active-total  active coil count in actor/action order
--n-angles        boundary/radii angle count
--checkpoint      optional PyTorch actor state_dict checkpoint
```

## Programmatic Use

The experiment config loader is:

```python
from tokamak_rl.config import load_experiment_config
```

It returns an `ExperimentConfig` with environment, reward, randomization, and resolved referenced config paths.

The environment entry point is:

```python
from tokamak_rl.env import EnvConfig, TokamakRLEnv
```

The environment wraps the simulator bridge, returns Gym-style reset/step tuples, maps normalized policy actions to physical active-coil current derivatives, builds observations from measured channels, and computes rewards from true simulator channels.

The action helper is `tokamak_rl.actions.ActionScaler`. It maps normalized actions in `[-1, 1]` to physical derivative commands using the simulator derivative scale.

The observation interface is based on `tokamak_rl.observations.ObservationSchema` and `tokamak_rl.observations.builder.build_observation`.

The observation schema version is `v1`. Its dimension is:

```text
5 + 2 * n_active_total + 3 * n_angles
```

The observation fields are phase, boundary-valid flag, measured Ip, Ip reference, Ip error, measured active currents, measured radii, reference radii, radii error, and previous normalized action.

Training presets now use observation schema `v2`, which appends configurable future target preview fields for Ip and boundary radii. Legacy configs without `sim.observation.version: v2` continue to use schema `v1`.

Episode termination rules are configured under `sim.termination`. The environment returns stable `termination_reason` names and a separate `termination_detail` string for simulator or rule-specific details.

The current randomization object is `tokamak_rl.randomization.DomainRandomizer`. Experiment `randomization_config` files are loaded, passed into the environment by the training and evaluation CLIs, recorded in reset/rollout metadata, and converted into tokamak-sim runtime sensor/actuator noise settings. Supported randomization is limited to simulator-backed measurement and actuation perturbations; plant-parameter randomization remains absent until tokamak-sim exposes explicit plant hooks.

The reward implementation is `tokamak_rl.rewards.JointCurrentBoundaryReward`. It computes plasma-current tracking, target-point-to-boundary shape tracking, action magnitude penalty, action-change penalty, optional current and derivative margin penalties, and termination penalty. Shape tracking follows the TCV-style target-point objective: reference radii are converted to target boundary points, then each target point is scored by its shortest distance to the true plasma boundary polyline. Experiment `reward_config` files are loaded and passed into the environment by the training and evaluation CLIs.

Training runs with `output_dir` stream scalar reward components into `reward_components.csv` with `step`, `env_index`, `episode`, `component`, and `value` columns.

The actor network is `tokamak_rl.networks.FeedForwardActor` configured by `ActorConfig`. The deterministic actor path returns bounded actions in `[-1, 1]`.

The critic networks are `tokamak_rl.networks.FeedForwardQCritic` and `tokamak_rl.networks.RecurrentQCritic`.

Policy export is handled by `tokamak_rl.export.export_actor`, with NumPy runtime loading through `tokamak_rl.export.load_numpy_actor`.

## Training And Evaluation

The repository currently contains a simple actor-critic trainer and a TCV-style recurrent-critic trainer.

The simple trainer uses `ReplayBuffer`, `SimpleTrainerConfig`, and `train_simple_actor_critic`. It stores individual transitions in a fixed-size circular replay buffer and uses feedforward twin critics.

The TCV-style trainer uses `EpisodeReplayBuffer`, sequence replay, recurrent critics, `TCVStyleTrainerConfig`, and `train_tcv_style_actor_critic`. It stores complete episodes, samples padded fixed-length chunks, trains twin recurrent critics, and updates the feedforward actor through sampled-action MPO policy improvement with KL-constrained actor fitting. Trainer `metrics.json` includes `tracking_diagnostics`, `eval_tracking_diagnostics`, MPO metrics, algorithm identity, and `run_metadata` for target diagnostics and artifact provenance. `tests/test_synthetic_training_contract.py` protects both simple and TCV-style real synthetic-reference training smoke paths.

Both trainers support `checkpoint_interval_steps`, write numbered `step_XXXXXXXX.pt` checkpoints at that cadence, keep `latest.pt`, and select `best.pt` by deterministic evaluation mean return. Checkpoint payloads include the actor, critics, target networks, optimizer states, counters, configuration, run metadata, and RNG metadata. Trainer `metrics.json` records the final, latest, and best checkpoint paths.

Both trainers resolve `device` through `cpu`, `cuda`, or `auto`. Requesting `cuda` fails clearly when CUDA is unavailable; `auto` uses CUDA when available and otherwise CPU. Trainer `metrics.json` records requested/actual device, CUDA availability, GPU name, PyTorch version, CUDA version, and throughput metrics including elapsed time, actor inference time, environment step time, replay sampling time, learner time, evaluation time, environment steps per second, learner updates per second, and update-to-data ratio. Synchronous multi-env collection batches actor inference before stepping environments. With `--process-envs`, each training environment is a process-owned `TokamakRLEnv` worker, so simulator stepping can run in parallel CPU processes while learner inference/updates stay in the main process. The TCV-style trainer exposes `updates_per_episode`, `updates_per_env_step`, and `max_learner_catchup_updates` to control update-to-data ratio.

The TCV-style trainer is the main training direction. Its implemented algorithm identity is `tcv_mpo_recurrent_actor_critic_v1`: deployable feedforward stochastic actor, twin recurrent Q critics used during training, complete-episode replay, masked sequence chunks, sampled-action MPO policy improvement, KL-constrained actor fitting, and deterministic actor mean for evaluation/export.

Rollout evaluation uses `tokamak_rl.evaluation.rollouts.run_rollout_evaluation` and writes summary JSON, episode metrics CSV, and rollout NPZ artifacts.

The evaluation CLI prints the output directory and artifact paths after writing them.

## Export

Deterministic actor export writes:

```text
policy_weights.npz
schema.json
normalization.json
metadata.json
```

The exported NumPy runtime evaluates the deterministic mean path of the actor and validates exported schema expectations when provided.

## Documentation

- [Repository Layout](docs/repository-layout.md)
- [Workflows](docs/workflows.md)
- [Architecture](docs/architecture.md)

## Notes

Training dependencies, replay buffers, policy networks, reward definitions, random trajectory generation, evaluation scripts, and exported policy artifacts belong in this repository.

Simulator physics, machine configuration loading, physical boundary extraction, run artifacts, and the bridge API belong in `tokamak-sim`.
