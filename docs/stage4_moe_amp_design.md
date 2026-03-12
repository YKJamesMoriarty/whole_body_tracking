# Stage4: Frozen Experts + Router MLP + AMP Design Notes

## 1. Target Architecture

Stage4 policy = `MoEActorCritic`:

- Frozen experts: 7 pre-trained mimic policies from `basic_model/`.
- Trainable router: MLP outputs softmax weights over 7 experts.
- Final action: weighted sum of expert actions.
- Trainable critic: standard PPO value head.

Current implementation files:

- `source/whole_body_tracking/whole_body_tracking/learning/moe_actor_critic.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/stage4/skill_registry.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/stage4/amp_discriminator.py`
- `source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`

## 2. Stage4 Training Path in `train.py`

New Stage4 switches:

- `--stage4_moe`
- `--frozen_model_dir`
- `--router_hidden_dims`
- `--router_init_noise_std`
- `--hit_radius_start`, `--hit_radius_end`, `--hit_curriculum_window`
- `--hit_curriculum_success_threshold`, `--hit_radius_shrink_factor`
- `--target_visible_time_min`, `--target_visible_time_max`
- `--stage4_episode_length_s`
- `--enable_amp_reward`, `--amp_disc_bundle_path`, `--amp_reward_weight`
- `--amp_disc_obs_mode` (`legacy_simple` / `mimickit_like`)
- `--amp_obs_history_steps` (通常与 MimicKit 的 `num_disc_obs_steps` 对齐)
- `--enable_router_diversity_reward`, `--router_diversity_weight`

Behavior when `--stage4_moe` is enabled:

- Uses custom policy class `whole_body_tracking.learning.moe_actor_critic.MoEActorCritic`.
- Loads frozen experts from a deterministic skill registry.
- Disables `motion_completed` termination.
- Keeps short target-visible window (0.3~0.8s configurable).
- Enables hit-radius curriculum in `MotionCommand`.
- Disables near reward + single-motion mimic rewards by default.
- AMP reward can be turned on/off for ablation.
- Router anti-collapse reward can be turned on/off for ablation.

## 3. AMP Core Concepts (for your questions)

### Q1: AMP needs positive samples?

Yes.

- Positive samples: reference motion snippets from motion library.
- Negative samples: current policy rollouts.
- Discriminator is a binary classifier between reference and policy motion.
- Style reward is derived from discriminator output (high when behavior looks like reference).

### Q2: Why AMP should be integrated into `whole_body_tracking`?

Because final router PPO runs inside IsaacLab (`whole_body_tracking`), style reward must be computed online in the same env loop.

If AMP is only trained in MimicKit but discriminator input preprocessing is not reproduced in IsaacLab, reward distribution will mismatch and become unreliable.

### Q3: AMP I/O for Stage4 router training

- Input: normalized `disc_obs` (history state features; dimension depends on env config and `num_disc_obs_steps`).
- Output: one logit per sample.
- Reward transform: typically from discriminator probability/logit to scalar style reward.

For compatibility, you must keep these identical between MimicKit and IsaacLab:

- body/joint ordering
- `global_obs`, `root_height_obs`
- `num_disc_obs_steps`
- disc observation normalization stats

## 4. How to Train AMP in `MimicKit_copy` with Your 7 Motions

### 4.1 Build a multi-motion yaml from your folder

```bash
cd ~/Desktop/IROS/whole_body_tracking
python scripts/amp/build_mimickit_motion_yaml.py \
  --motion_dir iros_motion/amp \
  --output_yaml /tmp/iros_amp_motions.yaml
```

### 4.2 Point MimicKit env config to this yaml

Edit `~/Desktop/IROS/MimicKit_copy/data/envs/amp_g1_env.yaml`:

- `motion_file: "/tmp/iros_amp_motions.yaml"`
- keep `num_disc_obs_steps` fixed (must match future IsaacLab-side implementation).

### 4.3 Train AMP

```bash
cd ~/Desktop/IROS/MimicKit_copy
python mimickit/run.py --arg_file args/amp_g1_args.txt
```

## 5. Exporting Discriminator for Later Integration

Use the helper script:

```bash
cd ~/Desktop/IROS/whole_body_tracking
python scripts/amp/export_mimickit_disc.py \
  --input_ckpt ~/Desktop/IROS/MimicKit_copy/output/hand_boxing_6/model_1.pt \
  --output_bundle /tmp/amp_disc_bundle.pt
```

Exported bundle includes:

- `disc_state_dict`
- `_disc_obs_norm` stats (`count/mean/std`)
- `disc_obs_dim`

## 6. Current Status and Remaining Gap

Implemented now:

1. Stage4 MoE routing + hit-radius curriculum.
2. AMP reward switch (`--enable_amp_reward`) and discriminator runtime inference path.
3. Export/import toolchain for MimicKit discriminator bundle.

Current notes:

1. Runtime now supports two modes:
   - `legacy_simple`: 旧版简化特征
   - `mimickit_like`: 按 MimicKit `compute_disc_obs` 结构构造（历史帧、ref-root、tan-norm、速度项）
2. 建议实验时将 `--amp_obs_history_steps` 与 MimicKit `num_disc_obs_steps` 设为一致（常见为 10）。
3. 最终实验建议关闭 `amp_allow_obs_dim_mismatch`，确保维度不匹配直接报错而非 pad/trim。
