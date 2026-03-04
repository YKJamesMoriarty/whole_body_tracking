# Whole Body Tracking: Two-Stage Humanoid Robot Control

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

本项目实现了一个**两阶段**的人形机器人控制框架，用于训练 Unitree G1 机器人执行拳击动作：

- **Stage 1: Motion Imitation (动作模仿)** - 基于 [BeyondMimic](https://beyondmimic.github.io/) 框架，训练机器人跟踪参考动作
- **Stage 2: Task-Oriented RL (任务导向强化学习)** - 在 Stage 1 基础上，训练机器人击打目标点

### 两阶段训练流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Motion Imitation (BeyondMimic)                                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│  目标: 学习参考动作的运动模式                                              │
│  输入: 参考动作数据 (cross_right_normal_body.npz)                        │
│  输出: 能够稳定执行出拳动作的策略                                          │
│  奖励: Mimic 奖励 (位置、朝向、速度误差)                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Task-Oriented RL                                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│  目标: 学习精准击打目标                                                   │
│  输入: Stage 1 预训练权重 + 固定点附近随机目标                              │
│  输出: 能够击打任意位置目标的策略                                          │
│  奖励: hit前(near+hit) + hit后(回位) + 全程mimic                            │
│  观测: 开局短时可见目标，随后隐藏，学习“看一眼后闭眼击打”                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 参考项目

- [BeyondMimic](https://beyondmimic.github.io/) - Stage 1 动作模仿框架
- [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) - Sim-to-real 部署

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# Option 2: HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- Pull the robot description files from GCS

```bash
# Enter the repository
cd whole_body_tracking
# Rename all occurrences of whole_body_tracking (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Motion Preprocessing & Registry Setup

In order to manage the large set of motions we used in this work, we leverage the WandB registry to store and load
reference motions automatically.
Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  Unitree's dataset

    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
    - Balance motions are from [HuB](https://hub-robot.github.io/)


- Log in to your WandB account; access Registry under Core on the left. Create a new registry collection with the name "
  Motions" and artifact type "All Types".


- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics,

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

This will automatically upload the processed motion file to the WandB registry with output name {motion_name}.

- Test if the WandB registry works properly by replaying the motion in Isaac Sim:

```bash
python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

#### Stage 1: Motion Imitation

基于 BeyondMimic 训练机器人跟踪参考动作：

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
  --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

#### Stage 2: Task-Oriented RL

在 Stage 1 预训练权重基础上，训练机器人击打目标：

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-organization}-org/wandb-registry-motions/cross_right_normal_body:v0 \
  --wandb_checkpoint_path {stage1-wandb-run-path} \
  --headless --logger wandb --num_envs 128 --max_iterations 400000
```

**Stage 2 关键特性：**

| 特性 | 说明 |
|------|------|
| **固定目标 + 随机扰动** | 每个 episode 在固定中心点附近 (`±0.2m`) 采样目标 |
| **短时可见观测** | 目标仅在 episode 开始 `0.3-0.8s` 可见，之后观测隐藏为 `(0,0,-10)` |
| **两阶段奖励** | hit 前使用 `near + hit` 任务奖励；hit 后关闭任务奖励，仅保留 mimic + 回位奖励 |
| **Hit 有效部位约束** | 仅四个末端可触发 hit：左右手腕、左右脚踝 |
| **单动作单回合** | 通过 `motion_completed` 终止条件，使一个裁剪动作对应一个 episode |

**Stage 2 奖励配置：**

```
任务奖励:
  - effector_target_hit:      12.0  (有效 hit 脉冲奖励)
  - effector_target_near:      8.0  (靠近目标进展奖励，hit 前有效)

Hit 后奖励:
  - post_hit_return_to_start:  4.0  (鼓励回到起始位置)

Mimic 奖励 (全程保留):
  - motion_anchor_pos/ori
  - motion_body_pos/ori
  - motion_body_lin/ang_vel

正则化惩罚 (Stage 1 & 2 共用):
  - action_rate_l2:     -0.1  (动作平滑，惩罚动作变化过大)
  - joint_limit:       -10.0  (关节超限惩罚)
  - undesired_contacts: -0.1  (非预期接触惩罚)
```

### Policy Evaluation

- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 \
  --wandb_path={wandb-run-path} \
  --registry_name {your-organization}-org/wandb-registry-motions/{motion_name}:latest \
  --eval_strict
```

The WandB run path can be located in the run overview. It follows the format {your_organization}/{project_name}/ along
with a unique 8-character identifier. Note that run_name is different from run_path.

## Code Structure

Below is an overview of the code structure for this repository:

### MDP 核心模块

**`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**

此目录包含 MDP (Markov Decision Process) 的核心组件实现:

#### `commands.py` - 动作命令与目标管理

**核心类:**
- `MotionLoader`: 从 NPZ 文件加载参考动作数据 (关节位置/速度、body 位姿/速度)
- `MotionCommand`: 动作命令管理器，负责参考动作播放和目标位置管理

**Stage 1 功能 (Motion Imitation):**
- 参考动作数据加载与插值
- 机器人当前状态与参考动作的误差计算
- 自适应采样 (Adaptive Sampling): 根据失败率调整初始化分布
- 初始状态随机化: 随机选择动作帧作为 episode 起点

**Stage 2 功能 (Task-Oriented RL):**
- **固定点随机采样**: `_resample_target_positions()` 在固定中心点附近采样真实目标位置
- **短时可见机制**: `target_visible_time_range_s` 控制开局可见窗口，之后观测隐藏
- **Hit 分阶段开关**: `check_hit()` 后关闭任务奖励，仅保留 mimic 与 post-hit 回位奖励
- **有效末端约束**: 仅四个末端 (双手/双脚) 触发 hit 判定
- **可维护的末端映射**: 根据 motion 名自动映射 active effector one-hot
- **可视化**: 目标小球和引导球线框调试显示

**关键配置 (`MotionCommandCfg`):**
```python
hit_distance_threshold = 0.06     # Hit 距离阈值 (米)
fixed_target_local_pos = (0.625, 0.0, 0.20)
target_randomization_local_range = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2)}
target_visible_time_range_s = (0.3, 0.8)
hidden_target_obs_local = (0.0, 0.0, -10.0)
```

---

#### `rewards.py` - 奖励与惩罚函数

**Stage 1 Mimic 奖励:**
- `motion_global_anchor_position_error_exp()`: 锚点 (torso) 位置跟踪
- `motion_global_anchor_orientation_error_exp()`: 锚点朝向跟踪
- `motion_relative_body_position_error_exp()`: 14 个 body 位置跟踪
- `motion_relative_body_orientation_error_exp()`: 14 个 body 朝向跟踪
- `motion_global_body_linear_velocity_error_exp()`: 线速度跟踪
- `motion_global_body_angular_velocity_error_exp()`: 角速度跟踪

**通用正则化惩罚 (Stage 1 & Stage 2 共用):**
- `action_rate_l2()`: 动作变化率惩罚，鼓励平滑动作 (权重: -0.1)
- `joint_pos_limits()`: 关节角度超限惩罚 (权重: -10.0)
- `undesired_contacts()`: 非预期接触惩罚，惩罚除脚踝和手腕外的身体部位触地 (权重: -0.1)

**Stage 2 任务奖励:**
- `effector_target_hit()`: 命中脉冲奖励（仅首次 hit）
- `effector_target_near()`: 距离进展奖励（仅 hit 前生效）
- `post_hit_return_to_start_exp()`: hit 后回到起始位置奖励


**奖励数学形式:**
```
Mimic 奖励: reward = exp(-error² / std²)
Hit 奖励: reward = 1.0 (脉冲)
回位奖励: reward = exp(-||p_xy - p_start_xy||² / std_xy²)
动作平滑: penalty = -0.1 × ||a_t - a_{t-1}||²
```

---

#### `observations.py` - 观测函数

**原始 Mimic 观测 (160 维):**
- `generated_commands`: 参考动作关节位置/速度 (58 维)
- `motion_anchor_pos_b/ori_b`: 参考锚点相对位置/朝向 (9 维)
- `base_lin_vel/ang_vel`: 基座速度 (6 维)
- `joint_pos_rel/vel_rel`: 关节位置/速度 (58 维)
- `last_action`: 上一步动作 (29 维)

**Stage 2 任务导向观测 (28 维):**
- `target_relative_position()`: 目标在机器人局部坐标系中的位置 (3 维，短时可见后隐藏)
- `target_relative_velocity()`: 目标相对速度 (3 维, 当前为零)
- `strikes_left()`: 剩余攻击次数 (1 维, 预留)
- `time_left()`: 剩余时间 (1 维, 预留)
- `active_effector_one_hot()`: 活跃攻击肢体 (4 维: [左手, 右手, 左脚, 右脚], 自动映射)
- `skill_type_one_hot()`: 16 维全 0 (保留通道，当前不使用)

**观测维度统计:**
| 网络 | Stage 1 | Stage 2 |
|------|---------|---------|
| Policy (Actor) | 160 维 | 188 维 |
| Critic (Value) | 286 维 | 314 维 |

---

#### `events.py` - 域随机化

**启动时随机化 (`mode="startup"`):**
- `randomize_rigid_body_material()`: 地面摩擦系数随机化
- `randomize_joint_default_pos()`: 关节默认位置偏移 (模拟校准误差)
- `randomize_rigid_body_com()`: 躯干质心偏移

**间隔随机化 (`mode="interval"`):**
- `push_robot()`: 随机外力推动 (1-3秒间隔)

---

#### `terminations.py` - 终止条件

**Stage 1 终止条件 (已在 Stage 2 中禁用):**
- `bad_anchor_pos()`: 锚点位置偏差过大
- `bad_anchor_ori()`: 锚点朝向偏差过大
- `bad_motion_body_pos()`: 四肢位置偏差过大

**Stage 2 终止条件:**
- `motion_completed()`: 动作参考轨迹播放完成（单动作单回合）
- `time_out()`: Episode 超时 (兜底)
- `robot_falling()`: 摔倒检测 (高度<0.3m 或 倾斜>55°)

---

### 环境配置

**`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**

环境 MDP 的完整配置:

| 配置类 | 说明 |
|--------|------|
| `MySceneCfg` | 场景配置: 地形、光照、接触传感器 |
| `CommandsCfg` | 动作命令配置: 目标采样范围、Hit 参数 |
| `ActionsCfg` | 动作空间: 29 DOF 关节位置控制 |
| `ObservationsCfg` | 观测配置: Policy (188维) + Critic (314维) |
| `RewardsCfg` | 奖励配置: 任务奖励 + Mimic 奖励 + 惩罚项 |
| `TerminationsCfg` | 终止条件: motion_completed + 超时兜底 + 摔倒检测 |
| `EventCfg` | 域随机化配置 |

**关键环境参数:**
```python
decimation = 4          # 控制频率 = 200Hz / 4 = 50Hz
sim.dt = 0.005          # 仿真步长 5ms
episode_length_s = 10.0 # Episode 超时兜底
motion_completed = True # 一个动作长度对应一个 episode
num_envs = 4096         # 并行环境数
```

---

### PPO 配置

**`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**

PPO 算法超参数配置:
- 网络架构: Actor MLP (256×128×64) + Critic MLP (256×128×64)
- 学习率: 1e-4 (或根据需要调整)
- Batch size, GAE lambda, Clip range 等

---

### 机器人配置

**`source/whole_body_tracking/whole_body_tracking/robots`**

Unitree G1 机器人特定配置:
- URDF/USD 模型路径
- 关节刚度/阻尼计算
- 动作缩放 (Action Scale)
- Armature 参数
- 被跟踪的 14 个 body 名称列表

---

### 脚本

**`scripts`**

| 脚本 | 功能 |
|------|------|
| `csv_to_npz.py` | 将重定向的 CSV 动作数据转换为 NPZ 格式，上传到 WandB |
| `replay_npz.py` | 在 Isaac Sim 中回放 NPZ 动作数据 |
| `upload_npz.py` | 手动上传 NPZ 到 WandB Registry |
| `rsl_rl/train.py` | 训练脚本，支持 Stage 1/2 训练 |
| `rsl_rl/play.py` | 策略评估与可视化 |

## Unitree G1 29 DOF Joint Names

以下为本项目使用的 Unitree G1 机器人 29 个关节自由度（DOF）名称：

1. left_hip_pitch_joint
2. right_hip_pitch_joint
3. waist_yaw_joint
4. left_hip_roll_joint
5. right_hip_roll_joint
6. waist_roll_joint
7. left_hip_yaw_joint
8. right_hip_yaw_joint
9. waist_pitch_joint
10. left_knee_joint
11. right_knee_joint
12. left_shoulder_pitch_joint
13. right_shoulder_pitch_joint
14. left_ankle_pitch_joint
15. right_ankle_pitch_joint
16. left_shoulder_roll_joint
17. right_shoulder_roll_joint
18. left_ankle_roll_joint
19. right_ankle_roll_joint
20. left_shoulder_yaw_joint
21. right_shoulder_yaw_joint
22. left_elbow_joint
23. right_elbow_joint
24. left_wrist_roll_joint
25. right_wrist_roll_joint
26. left_wrist_pitch_joint
27. right_wrist_pitch_joint
28. left_wrist_yaw_joint
29. right_wrist_yaw_joint
