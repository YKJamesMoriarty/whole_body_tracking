# Whole Body Tracking: Two-Stage Humanoid Robot Boxing/Kicking Training

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

---

## 项目概述 (Project Overview)

本项目实现了一套**两阶段**的人形机器人攻击动作训练框架，用于训练 **Unitree G1** 机器人执行精准的拳击/踢击动作并击打固定目标。

**核心贡献：**
- 基于 [BeyondMimic](https://beyondmimic.github.io/) 框架扩展了第二阶段任务导向强化学习
- 设计了"进展奖励"（Progress Reward）机制，通过历史最近距离跟踪避免蹭分
- 采用固定目标 + 冷却期收手设计，鼓励机器人打完就收，形成完整攻击周期
- 支持多种攻击技能（右直拳 Cross、高位鞭腿 Roundhouse、正蹬 Front Kick）

**参考项目：**
- [BeyondMimic](https://beyondmimic.github.io/) — Stage 1 动作模仿框架
- [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller) — Sim-to-real 部署

---

## 两阶段训练流程 (Two-Stage Training Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Motion Imitation (BeyondMimic 框架)                           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│  目标: 让机器人学会模仿参考运动数据，掌握出拳/踢击的基本动作模式              │
│  输入: 参考动作 NPZ 文件 (含关节角度/速度、body 位姿/速度)                  │
│  输出: 能稳定执行攻击动作的策略网络 (预训练权重)                            │
│  奖励: 全 Mimic 奖励 (锚点位置/朝向 + 14个body位置/朝向 + 速度)             │
│  终止: anchor 误差过大 / 四肢误差过大 (严格跟踪约束)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                     Stage 1 预训练权重 (--wandb_checkpoint_path)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Task-Oriented RL (任务导向强化学习)                            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│  目标: 在保持动作质量的前提下，学习精准击打固定目标点                        │
│  输入: Stage 1 权重 + 目标球位置观测 (新增 28 维任务导向观测)               │
│  输出: 能用攻击肢体精准击打指定位置目标的策略                               │
│  奖励: Hit 奖励 + 进展奖励 + 速度方向奖励 + 弱化 Mimic 奖励 + 姿态惩罚      │
│  终止: 摔倒检测 (高度<0.25m 或 倾斜>55°) + 超时 (20s)                      │
│                                                                         │
│  【课程学习 - 已注释，计划后期启用】                                        │
│  设计: 目标采样范围随训练进展从固定点扩大到完整区间                           │
│  等级: 0 → 0.25 → 0.5 → 0.75 → 1.0 (共5级，基于滑动窗口 Hit 率升级)       │
│  暂不使用原因: 不同技能适合打到的高度不同，采样区间设置需要精细调校            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 支持的攻击技能 (Supported Skills)

| Branch | 技能名 | 攻击肢体 | 参考动作文件 | effector_body_name | skill_type index |
|--------|--------|----------|-------------|-------------------|-----------------|
| `basic_attack` | Cross (右直拳) | 右手腕 | `cross_right_normal_body.npz` | `right_wrist_yaw_link` | 0 |
| `basic_attack` | Roundhouse High (右高位鞭腿) | 右脚踝 | `roundhouse_right_fast_high.npz` | `right_ankle_roll_link` | 3 |
| `basic_attack_right_frontkick` | Front Kick (右脚正蹬) | 右脚踝 | `frontkick_right_normal_body.npz` | `right_ankle_roll_link` | 4 |

**当前活跃分支: `basic_attack_right_frontkick`**
- 攻击肢体: 右脚踝 (`right_ankle_roll_link`)
- 目标固定点: `(0.5, 0.0, -0.3)` (相对于参考动作第一帧 Pelvis 局部坐标系)
  - x=0.5m (前方50cm), y=0 (居中), z=-0.3m (骨盆以下30cm)
- skill_type one-hot: 索引4 对应 `frontkick_right_normal_body`

---

## 安装 (Installation)

1. 安装 Isaac Lab v2.1.0（参考 [官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)，推荐 conda）

2. 克隆本仓库（独立于 IsaacLab 目录）：
```bash
git clone git@github.com:HybridRobotics/whole_body_tracking.git
cd whole_body_tracking
```

3. 下载机器人描述文件：
```bash
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

4. 安装本库：
```bash
python -m pip install -e source/whole_body_tracking
```

---

## 参考动作数据准备 (Motion Data Preparation)

### 数据来源
- Unitree 重定向的 LAFAN1 数据集: [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
- 踢击动作 (Sidekicks): [KungfuBot](https://kungfu-bot.github.io/)
- 格斗动作: 通过运动捕获设备录制并重定向到 G1 机器人

### 数据处理流程
```bash
# 1. 将 CSV 格式参考动作转换为 NPZ（含正向运动学计算的 body 位姿/速度/加速度）
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 \
  --output_name {motion_name} --headless
# 转换后自动上传到 WandB Registry

# 2. 在 Isaac Sim 中回放验证
python scripts/replay_npz.py \
  --registry_name={your-org}-org/wandb-registry-motions/{motion_name}
```

### NPZ 文件内容
| 字段 | 形状 | 含义 |
|------|------|------|
| `joint_pos` | `(T, 29)` | 关节位置 (rad) |
| `joint_vel` | `(T, 29)` | 关节速度 (rad/s) |
| `body_pos_w` | `(T, N, 3)` | N 个 body 在世界坐标系的位置 |
| `body_quat_w` | `(T, N, 4)` | N 个 body 在世界坐标系的四元数 |
| `body_lin_vel_w` | `(T, N, 3)` | N 个 body 的线速度 |
| `body_ang_vel_w` | `(T, N, 3)` | N 个 body 的角速度 |
| `fps` | scalar | 采样频率 (通常 30 fps) |

---

## 训练指令 (Training Commands)

### Stage 1: Motion Imitation
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-org}-org/wandb-registry-motions/{motion_name}:v0 \
  --headless --logger wandb \
  --log_project_name {project_name} \
  --run_name {run_name}
```

### Stage 2: Task-Oriented RL
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-org}-org/wandb-registry-motions/{motion_name}:v0 \
  --wandb_checkpoint_path {stage1-wandb-run-path} \
  --headless --logger wandb \
  --num_envs 4096 \
  --max_iterations 400000 \
  --log_project_name {project_name} \
  --run_name {run_name}
```

### 评估/可视化 (Play)
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 \
  --num_envs=2 \
  --wandb_path={wandb-run-path}
```

---

## 核心架构 (Core Architecture)

### 文件结构

```
source/whole_body_tracking/whole_body_tracking/tasks/tracking/
├── tracking_env_cfg.py          # 主配置文件 (场景/观测/奖励/终止/事件)
├── mdp/
│   ├── commands.py              # MotionLoader + MotionCommand (核心状态机)
│   ├── rewards.py               # 所有奖励/惩罚函数
│   ├── observations.py          # 观测函数
│   ├── terminations.py          # 终止条件
│   └── events.py                # 域随机化
└── config/g1/
    ├── flat_env_cfg.py          # G1 机器人特化配置 (14 个 body 名称)
    └── agents/rsl_rl_ppo_cfg.py # PPO 超参数

scripts/
├── csv_to_npz.py    # 参考动作格式转换 + 上传 WandB
├── replay_npz.py    # Isaac Sim 中回放参考动作
├── upload_npz.py    # 手动上传 NPZ 到 WandB Registry
└── rsl_rl/
    ├── train.py     # 训练入口
    └── play.py      # 评估/可视化入口
```

---

## 详细模块说明

### 1. `commands.py` — 核心状态机

#### `MotionLoader` 类
从 NPZ 文件加载参考动作数据。只加载配置中指定的 `body_names` 对应的数据。

#### `MotionCommand` 类
**最核心的类**，同时管理：
- 参考动作数据播放（帧索引递增）
- 目标球位置管理
- Hit 检测与冷却期状态机
- 进展奖励相关状态
- 可视化 marker

**Stage 2 关键状态变量：**

| 变量 | 类型 | 含义 |
|------|------|------|
| `target_pos_w` | `(N, 3)` | 目标球世界坐标（Hit 后冷却期设为 z=-10） |
| `task_rewards_enabled` | `(N,) bool` | 任务奖励是否生效（冷却期内为 False） |
| `hit_resample_timer` | `(N,)` | 冷却期剩余时间（秒，>0 表示在冷却中） |
| `min_distance_to_target` | `(N,)` | 当前 Hit 周期内历史最近距离（进展奖励用） |
| `cumulative_hit_count` | `(N,)` | 当前 episode 内累积 Hit 次数 |
| `has_entered_guidance_sphere` | `(N,) bool` | 是否进入过引导球 |
| `_spawn_target_timer` | `(N,)` | Episode 初始化延迟计时器 |

**Stage 2 Hit 检测流程（每个 step）：**
```
effector_target_hit() 被调用 (rewards.py)
    │
    ├── command.update_hit_resample_timer()
    │   ├── 冷却计时器 -= step_dt
    │   ├── 若冷却结束 → _resample_target_positions() → 任务奖励重新生效
    │   ├── spawn 计时器 -= step_dt
    │   └── 若 spawn 延迟结束 → _resample_target_positions()
    │
    ├── command.check_hit()
    │   ├── 计算 effector 到 target 距离
    │   ├── Hit 条件: distance < hit_distance_threshold (0.1m) AND task_rewards_enabled
    │   └── Hit 成功后:
    │       ├── hit_resample_timer = hit_resample_delay (1.8s)
    │       ├── task_rewards_enabled = False
    │       ├── target_pos_w z 设为 -10 (地下，防止误触发)
    │       └── cumulative_hit_count += 1
    │
    └── command.update_curriculum(hit_mask)
        └── 更新 WandB 监控指标 (hit_count, max_hit_count)
```

**Episode 重置流程：**
```
_resample_command() 调用
    ├── _adaptive_sampling(): 强制 time_steps = 0 (固定从第一帧开始)
    ├── _delayed_spawn_target(): 目标先隐藏到地下，启动 0.8s spawn 延迟计时器
    └── cumulative_hit_count = 0 (清零累积计数)
```

**目标位置计算（`_resample_target_positions`）：**
```python
# 固定局部偏移 (由 cfg.fixed_target_local_pos 配置)
local_pos = [fx, fy, fz]  # 例如 (0.5, 0.0, -0.3) for front kick

# 转换到世界坐标: 以参考动作第一帧 Pelvis 为基准
world_pos = reference_root_pos + quat_apply(reference_root_quat, local_pos)
# 加上各环境的空间偏移
world_pos += env_origins[env_ids]
```

#### `MotionCommandCfg` — 核心配置参数

```python
# 固定目标位置（相对于参考动作第一帧 Pelvis 局部坐标系）
# x > 0: 前方; y: 左右; z: 上下（正=骨盆以上）
fixed_target_local_pos: tuple = (0.5, 0.0, -0.3)  # 当前: 正蹬目标点

effector_body_name: str = "right_ankle_roll_link"  # 当前: 右脚踝

hit_distance_threshold: float = 0.1     # Hit 距离阈值 (米), 与目标球半径一致
hit_resample_delay: float = 1.8         # Hit 后冷却期 (秒)
spawn_target_delay: float = 0.8         # Episode 初始化延迟 (秒)
guidance_sphere_radius: float = 0.4     # 引导球半径 (米), 进展奖励触发范围
```

---

### 2. `rewards.py` — 奖励函数

#### 当前启用的奖励（Stage 2）

**任务奖励（Task Rewards）：**

| 奖励函数 | 权重 | 说明 |
|---------|------|------|
| `effector_target_hit` | +15.0 | 核心 Hit 奖励。距离 < 0.1m 且 task_rewards_enabled=True 时触发，返回 1.0 的脉冲奖励 |
| `effector_target_near` | +18.0 | 进展奖励。只在引导球(0.4m)范围内且比历史最近距离更近时给奖励，奖励量 = Δ距离 × 10。完美解决蹭分问题 |
| `effector_velocity_towards_target` | +1.5 | 速度方向奖励。effector 速度朝向目标时给奖励（cos θ 归一化到 [0,1]），只在引导球范围内生效 |
| `body_face_target` | 0.0 | 躯干朝向目标奖励（已禁用，weight=0） |

**动作质量奖励（Mimic Rewards，弱化版）：**

| 奖励函数 | 权重 | 说明 |
|---------|------|------|
| `motion_global_anchor_pos` | +1.5 | Anchor (torso_link) 位置跟踪，防止机器人漫游 |
| `motion_global_anchor_ori` | +1.5 | Anchor 朝向跟踪 |
| `motion_body_pos` | +3.0 | 全部 14 个 body 位置跟踪（含攻击肢体） |
| `motion_body_ori` | +3.0 | 全部 14 个 body 朝向跟踪 |
| `mimic_non_right_hand_body_pos` | +1.0 | 排除右腿三个 link 后的位置跟踪（攻击肢体自由探索目标） |
| `mimic_non_right_hand_body_ori` | +1.0 | 排除右腿三个 link 后的朝向跟踪 |
| `mimic_right_elbow_dof` | +1.0 | 右膝关节角度跟踪（`mimic_right_knee_dof_exp`，控制踢击高度） |
| `mimic_right_shoulder_roll_dof` | +1.0 | 右髋外展关节角度跟踪（`mimic_right_hip_roll_dof_exp`，控制鞭腿特征动作） |
| `motion_body_lin_vel` | +3.0 | 全身线速度跟踪 |
| `motion_body_ang_vel` | +3.0 | 全身角速度跟踪 |

**正则化惩罚：**

| 惩罚函数 | 权重 | 说明 |
|---------|------|------|
| `posture_unstable` | +1000.0（函数返回负值） | 身体倾斜 > ~40度时惩罚，tilt_threshold=0.234 |
| `action_rate_l2` | -0.1 | 动作平滑惩罚，抑制抖动 |
| `joint_limit` | -10.0 | 关节超限惩罚 |
| `undesired_contacts` | -0.1 | 非预期接触惩罚（排除脚踝和手腕） |

#### 注：排除攻击肢体的 body 列表（`right_hand_names` 参数）
当前配置（正蹬/鞭腿）排除右腿三个 link，让 task reward 驱动右腿运动：
```python
right_hand_names = ["right_hip_roll_link", "right_knee_link", "right_ankle_roll_link"]
```

#### 奖励数学形式
```
Mimic 奖励:      r = exp(-error² / std²)              # std 越小越严格
Hit 奖励:        r = 1.0 (脉冲，每次 Hit 触发一次)
进展奖励:        r = (min_hist_dist - current_dist) × scale  (仅当有进展时)
速度方向奖励:    r = 0.5 × (cos_θ + 1.0)              # 归一化到 [0, 1]
姿态惩罚:        p = -(tilt - threshold)               # tilt = 1 - dot(up_body, world_up)
动作平滑:        p = -||a_t - a_{t-1}||²
```

#### 已设计但未启用的奖励函数（在代码中存在，可参考）
- `pen_linger_in_hit_sphere()`: Hit 后冷却期内，手在小球内的常量惩罚（目前不需要）
- `rew_retract_from_target()`: Hit 后冷却期内，手远离目标的进展奖励（目前不需要）
- `effector_target_tracking_exp()`: effector 靠近参考动作指定位置的指数奖励（Stage 1 辅助）
- `mimic_right_leg_position/orientation_exp()`: 右腿末端强化 mimic（可选启用）

---

### 3. `observations.py` — 观测空间

#### Policy (Actor) 观测 — 188 维

**原始 Mimic 观测（160 维，与 Stage 1 完全相同）：**

| 观测 | 维度 | 内容 | 噪声 |
|------|------|------|------|
| `command` | 58 | 参考动作当前帧的关节位置 (29) + 关节速度 (29) | 无 |
| `motion_anchor_pos_b` | 3 | 参考 anchor 相对当前 anchor 的位置（局部坐标系） | ±0.25 |
| `motion_anchor_ori_b` | 6 | 参考 anchor 朝向（旋转矩阵前两列展开） | ±0.05 |
| `base_lin_vel` | 3 | 机器人基座线速度 | ±0.5 |
| `base_ang_vel` | 3 | 机器人基座角速度 | ±0.2 |
| `joint_pos` | 29 | 关节位置（相对默认位置） | ±0.01 |
| `joint_vel` | 29 | 关节速度（相对默认速度） | ±0.5 |
| `actions` | 29 | 上一步动作 | 无 |

**Stage 2 任务导向观测（28 维，新增）：**

| 观测 | 维度 | 内容 | 噪声 |
|------|------|------|------|
| `target_rel_pos` | 3 | 目标在机器人局部坐标系的位置 (x前后, y左右, z上下) | ±0.1，clip(-15,15) |
| `target_rel_vel` | 3 | 目标相对速度（当前始终为零向量，目标静止） | 无 |
| `strikes_left` | 1 | 当前 episode 内累积 Hit 次数（名称保留但语义改变） | 无 |
| `time_left` | 1 | 归一化剩余时间（当前始终为常数 1.0） | 无 |
| `active_effector` | 4 | 活跃攻击肢体 one-hot: [左手, 右手, 左脚, 右脚] | 无 |
| `skill_type` | 16 | 技能类型 one-hot（16 个技能槽位） | 无 |

**skill_type 索引对照表（16 维）：**
```
索引 0: r-Cross (右直拳)          → one_hot[:,0] = 1
索引 1: r-swing (右摆拳)
索引 2: roundhouse_right_normal_low (右低位鞭腿)
索引 3: roundhouse_right_fast_high (右高位鞭腿) → one_hot[:,3] = 1
索引 4: frontkick_right_normal_body (右脚正蹬)  → one_hot[:,4] = 1 (当前分支)
索引 5-15: 预留
```

**active_effector 索引对照表（4 维）：**
```
[左手, 右手, 左脚, 右脚]
右手: one_hot[:,1] = 1  (Cross)
右脚: one_hot[:,3] = 1  (Roundhouse/Front Kick，当前分支)
```

#### Critic (Privileged) 观测 — 314 维
在 Policy 观测基础上新增：
- `body_pos` (+42维): 14 个 body 在 anchor 局部坐标系的位置
- `body_ori` (+84维): 14 个 body 在 anchor 局部坐标系的朝向（旋转矩阵）
- 同样包含所有 28 维任务导向观测

#### 坐标系说明
- `target_rel_pos` 以 **机器人 Pelvis (Root) 为原点**，机器人前向为 X 轴
- 转换公式: `p_local = R_robot^{-1} × (p_target_world - p_robot_world)`
- 在冷却期/spawn 延迟期间，目标在地下 z=-10m，clip(-15,15) 防止极端值传播

---

### 4. `tracking_env_cfg.py` — 环境主配置

#### 关键环境参数

```python
decimation = 4           # 控制频率 = 200Hz / 4 = 50Hz
sim.dt = 0.005           # 物理仿真步长 5ms (200Hz)
episode_length_s = 20.0  # Episode 最大时长 20 秒
num_envs = 4096          # 并行训练环境数

# 域随机化 (startup)
static_friction: (0.3, 1.6)    # 地面静摩擦系数随机化
dynamic_friction: (0.3, 1.2)   # 地面动摩擦系数随机化
joint_default_pos: ±0.01 rad   # 关节零位偏移（模拟校准误差）
torso_com: x±2.5cm, y±5cm, z±5cm  # 躯干质心偏移

# 域随机化 (interval: 1~3s)
push_robot: 随机外力（速度扰动）
  x: ±0.5 m/s, y: ±0.5 m/s, z: ±0.2 m/s
  roll/pitch: ±0.52 rad/s, yaw: ±0.78 rad/s
```

#### 终止条件
| 条件 | 参数 | 说明 |
|------|------|------|
| `time_out` | 20s | Episode 超时 |
| `robot_falling` (高度) | `height < 0.25m` | Pelvis 高度过低 |
| `robot_falling` (倾斜) | `dot < 0.57 (~55°)` | 躯干过度倾斜 |

注：Stage 2 **不使用** Stage 1 的 mimic 终止条件（anchor 误差 / 四肢误差），允许机器人一定程度偏离参考动作。

---

### 5. G1 机器人配置 (`flat_env_cfg.py`)

#### 跟踪的 14 个 Body Links

```python
body_names = [
    "pelvis",                 # 0: 骨盆（参考根节点）
    "left_hip_roll_link",     # 1: 左髋
    "left_knee_link",         # 2: 左膝
    "left_ankle_roll_link",   # 3: 左踝
    "right_hip_roll_link",    # 4: 右髋（攻击肢体，鞭腿/正蹬时自由探索）
    "right_knee_link",        # 5: 右膝（攻击肢体）
    "right_ankle_roll_link",  # 6: 右踝（攻击肢体，Hit 检测点）
    "torso_link",             # 7: 躯干（Anchor）
    "left_shoulder_roll_link",# 8: 左肩
    "left_elbow_link",        # 9: 左肘
    "left_wrist_yaw_link",    # 10: 左腕
    "right_shoulder_roll_link",# 11: 右肩
    "right_elbow_link",       # 12: 右肘
    "right_wrist_yaw_link",   # 13: 右腕（Cross 时为 Hit 检测点）
]
anchor_body_name = "torso_link"  # 所有相对运动的基准 body
```

#### Unitree G1 29 DOF 关节列表

```
1. left_hip_pitch_joint      2. right_hip_pitch_joint    3. waist_yaw_joint
4. left_hip_roll_joint       5. right_hip_roll_joint     6. waist_roll_joint
7. left_hip_yaw_joint        8. right_hip_yaw_joint      9. waist_pitch_joint
10. left_knee_joint          11. right_knee_joint        12. left_shoulder_pitch_joint
13. right_shoulder_pitch_joint  14. left_ankle_pitch_joint  15. right_ankle_pitch_joint
16. left_shoulder_roll_joint 17. right_shoulder_roll_joint  18. left_ankle_roll_joint
19. right_ankle_roll_joint   20. left_shoulder_yaw_joint 21. right_shoulder_yaw_joint
22. left_elbow_joint         23. right_elbow_joint       24. left_wrist_roll_joint
25. right_wrist_roll_joint   26. left_wrist_pitch_joint  27. right_wrist_pitch_joint
28. left_wrist_yaw_joint     29. right_wrist_yaw_joint
```

---

### 6. PPO 算法配置 (`rsl_rl_ppo_cfg.py`)

```python
# 网络架构 (Actor-Critic 共享结构)
actor_hidden_dims  = [512, 256, 128]  # Actor MLP
critic_hidden_dims = [512, 256, 128]  # Critic MLP
activation = "elu"
init_noise_std = 1.0
empirical_normalization = True  # 在线归一化观测值

# PPO 超参数
num_steps_per_env = 24          # 每个 env 每次更新收集的步数
num_learning_epochs = 5         # 每批数据的训练轮数
num_mini_batches = 4            # 小批次数
learning_rate = 1e-3            # 初始学习率（adaptive schedule 自适应调整）
schedule = "adaptive"           # 根据 KL 散度自适应调整学习率
desired_kl = 0.01
clip_param = 0.2                # PPO clip 范围
value_loss_coef = 1.0
entropy_coef = 0.005
gamma = 0.99                    # 折扣因子
lam = 0.95                      # GAE lambda
max_grad_norm = 1.0
```

---

## 关键设计决策说明 (Key Design Decisions)

### 1. 进展奖励（Progress Reward）机制
**问题：** 手在目标附近停留可持续获得奖励（蹭分），导致机器人不打完就收。

**解决方案：** 只有当 effector 到目标的距离比**历史最近距离**更小时才给奖励：
```
reward = (min_historical_dist - current_dist) × scale  (仅当 current_dist < min_historical_dist)
```
效果：手停在任意位置 → 0奖励；手绕圈 → 0奖励；手向目标靠近 → 正奖励；打到目标 → 之后 Near 奖励自然归零。

### 2. Hit 后冷却期设计
**目的：** 鼓励机器人打完后跟随参考动作收手，而不是一直悬停。

**机制：**
- Hit 成功 → 1.8s 冷却期（task_rewards_enabled=False）
- 冷却期内：任务奖励（Hit/Near/velocity）全部失效，只有 Mimic 奖励驱动收手
- 目标球在冷却期内沉入地下（z=-10），进一步防止 Near 奖励蹭分
- 冷却结束 → 目标重新出现（同一固定位置），开始下一次攻击周期

### 3. Episode 初始化延迟 (Spawn Delay)
**问题：** Episode 重置时物理引擎给机器人施加随机速度噪声，瞬间可能意外触发 Hit。

**解决方案：** 重置后目标先放到地下（z=-10），等 0.8s 物理引擎稳定后再出现。

### 4. 固定目标点（不使用课程学习）
**现状：** 目前使用固定目标点训练，课程学习已注释。

**原因：** 课程学习（从中心点扩大采样范围）在不同技能中的合适高度区间不同，需要精细调校。目前先用固定点验证训练可行性。

**课程学习设计（已注释，待后期启用）：**
- 5 个等级：0.0, 0.25, 0.5, 0.75, 1.0（表示从初始点到最终采样区间的插值比例）
- 升级条件：滑动窗口（500步）内 Hit 率达到阈值（0.005）
- 初始采样范围 → 最终采样范围（x/y/z 区间分别线性插值）

### 5. Stage 2 始终从第一帧开始
**原因：** BeyondMimic 的自适应采样可能选到动作中间帧（如收拳阶段），此时引导机器人去打目标是不合理的。Stage 2 固定从 frame=0（标准站姿）开始每个 episode，确保训练一致性。

### 6. 攻击肢体自由探索
**设计：** `mimic_non_right_hand_body_pos/ori` 排除了攻击肢体对应的 link，让 Task Reward 完全驱动攻击肢体运动，Mimic 奖励只约束其他身体部分。

---

## WandB 监控指标说明 (Monitoring Metrics)

| 指标 | 含义 |
|------|------|
| `hit_count` | 当前 step 内 Hit 的环境数 |
| `max_hit_count` | 所有环境中累积 Hit 次数的最大值 |
| `effector_to_target_dist` | 攻击肢体到目标的距离（m） |
| `effector_speed` | 攻击肢体速度（m/s） |
| `error_anchor_pos` | Anchor（torso）位置跟踪误差 |
| `error_anchor_rot` | Anchor 朝向跟踪误差 |
| `error_body_pos` | 全身 body 位置跟踪误差均值 |
| `error_body_rot` | 全身 body 朝向跟踪误差均值 |
| `error_joint_pos` | 关节位置跟踪误差 |

---

## 调试工具 (Debugging Tools)

### play 模式实时速度日志
`commands.py` 的 `_debug_vis_callback` 中，当攻击肢体速度 > 0.5 m/s 时自动写入日志：
```bash
tail -f logs/hand_speed.log
# 格式: R: {speed:.2f} m/s | L: 0.00 m/s | Dist: {dist:.3f} m
```

### Isaac Sim 可视化
play 模式启用 `debug_vis=True` 时（默认开启）：
- **红色实心球**：目标位置（半径 = hit_distance_threshold）
- **浅绿色线框球**：引导球范围（radius = guidance_sphere_radius = 0.4m）
- **彩色坐标轴**：当前机器人各 body 位置（小坐标轴）与参考动作对应位置（大坐标轴）

### NaN 崩溃防护
系统有两道防护：
1. `observations.py`：`nan_to_num(nan=0, posinf=10, neginf=-10)` 源头拦截
2. `tracking_env_cfg.py`：ObsTerm 中 `clip=(-15, 15)` 兜底

---

## 已知限制与未来工作 (Known Limitations)

1. **课程学习未启用**：当前仅训练固定目标点，课程学习代码已预留（注释），待后期不同技能的高度区间调校后启用
2. **目标静止**：`target_relative_velocity` 始终返回零，不支持移动目标
3. **单技能训练**：每个分支固定训练一种技能（Cross/Roundhouse/FrontKick），不支持多技能联合训练
4. **`time_left` 未使用**：始终返回 1.0，未实现攻击窗口倒计时
5. **Pyright 类型错误**：部分框架接口的 `Tensor vs Sequence[int]`、`MISSING` 类型警告为 Isaac Lab 框架本身的问题，不影响运行

---

## 快速参考：修改目标点位置

目标点在 `commands.py` 的 `MotionCommandCfg.fixed_target_local_pos` 中配置：

```python
# 坐标系: 相对于参考动作第一帧 Pelvis 局部坐标系
# x > 0: 前方 (m)  |  y: 左右 (m, +为左)  |  z: 高度 (m, +为骨盆以上)

# 各技能建议目标点:
# Cross (右直拳):      (0.625, 0.0, 0.25)   # 前62.5cm, 居中, 骨盆上25cm (胸口高度)
# Roundhouse High:    (0.5, 0.0, 0.05)      # 前50cm, 居中, 骨盆上5cm
# Front Kick:         (0.5, 0.0, -0.3)      # 前50cm, 居中, 骨盆下30cm  ← 当前
```
