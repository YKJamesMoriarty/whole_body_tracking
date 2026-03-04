from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.stage4.skill_registry import STAGE4_SKILL_METRIC_NAMES

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        fixed_target_local_pos=(0.625, 0.0, 0.20),
        target_randomization_local_range={
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.2, 0.2),
        },
        target_visible_time_range_s=(0.3, 0.8),
        hidden_target_obs_local=(0.0, 0.0, -10.0),
        guidance_sphere_radius=0.4,
        guidance_sphere_follow_hit_radius=False,
        hit_distance_threshold=0.06,
        target_sphere_follow_hit_radius=False,
        router_metric_names=tuple(STAGE4_SKILL_METRIC_NAMES),
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor 策略网络的观测配置
        
        原始 Mimic 观测 (160 维):
            - command: 58 维 (关节位置 + 关节速度)
            - motion_anchor_pos_b: 3 维
            - motion_anchor_ori_b: 6 维
            - base_lin_vel: 3 维
            - base_ang_vel: 3 维
            - joint_pos: 29 维
            - joint_vel: 29 维
            - actions: 29 维
        
        新增任务导向观测 (28 维):
            - target_rel_pos: 3 维 (目标相对位置)
            - target_rel_vel: 3 维 (目标相对速度)
            - strikes_left: 1 维 (剩余攻击次数)
            - time_left: 1 维 (剩余时间)
            - active_effector: 4 维 (活跃肢体 one-hot)
            - skill_type: 16 维 (技能类型 one-hot)
        
        总计: 160 + 28 = 188 维
        """

        # =====================================================================
        # 原始 Mimic 观测 (请勿修改)
        # =====================================================================
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        # =====================================================================
        # Stage 2 任务导向观测
        # 使用课程学习动态采样的目标位置
        # =====================================================================
        
        # (1) 目标相对位置: 3 维
        # Stage 2: 使用 command.target_pos_w (课程学习采样的目标)
        target_rel_pos = ObsTerm(
            func=mdp.target_relative_position,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        
        # (2) 目标相对速度: 3 维
        # Stage 1: 设为零 (目标静止)
        target_rel_vel = ObsTerm(
            func=mdp.target_relative_velocity,
            params={"command_name": "motion"}
        )
        
        # (3) 剩余攻击次数: 1 维
        # Stage 1: 设为常数 1.0
        strikes_left = ObsTerm(
            func=mdp.strikes_left,
            params={"command_name": "motion"}
        )
        
        # (4) 剩余时间: 1 维
        # Stage 1: 设为常数 1.0
        time_left = ObsTerm(
            func=mdp.time_left,
            params={"command_name": "motion"}
        )
        
        # (5) 活跃攻击肢体 one-hot: 4 维 [左手, 右手, 左脚, 右脚]
        # Stage 1: 硬编码为右手 [0, 1, 0, 0]
        active_effector = ObsTerm(
            func=mdp.active_effector_one_hot,
            params={"command_name": "motion"}
        )
        
        # (6) 技能类型 one-hot: 16 维
        # 格式: [Jab, Cross, Hook, Uppercut, Backfist, Overhand,
        #        LowKick, MidKick, HighKick, FrontKick, SideKick, RoundhouseKick,
        #        Combo1, Combo2, 预留, 预留]
        # Stage 1: 硬编码为直拳
        skill_type = ObsTerm(
            func=mdp.skill_type_one_hot,
            params={"command_name": "motion"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic 价值网络的特权观测配置
        
        原始 Mimic 观测 (286 维):
            - 与 Policy 相同 + body_pos (42 维) + body_ori (84 维)
        
        新增任务导向观测 (28 维):
            - 与 Policy 相同
        
        总计: 286 + 28 = 314 维
        """
        
        # =====================================================================
        # 原始 Mimic 观测 (请勿修改)
        # =====================================================================
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        
        # =====================================================================
        # Stage 2 任务导向观测
        # Critic 也需要这些观测来准确估计价值函数
        # =====================================================================
        
        # (1) 目标相对位置: 3 维
        target_rel_pos = ObsTerm(
            func=mdp.target_relative_position,
            params={"command_name": "motion"}
        )
        
        # (2) 目标相对速度: 3 维
        target_rel_vel = ObsTerm(
            func=mdp.target_relative_velocity,
            params={"command_name": "motion"}
        )
        
        # (3) 剩余攻击次数: 1 维
        strikes_left = ObsTerm(
            func=mdp.strikes_left,
            params={"command_name": "motion"}
        )
        
        # (4) 剩余时间: 1 维
        time_left = ObsTerm(
            func=mdp.time_left,
            params={"command_name": "motion"}
        )
        
        # (5) 活跃攻击肢体 one-hot: 4 维
        active_effector = ObsTerm(
            func=mdp.active_effector_one_hot,
            params={"command_name": "motion"}
        )
        
        # (6) 技能类型 one-hot: 16 维
        skill_type = ObsTerm(
            func=mdp.skill_type_one_hot,
            params={"command_name": "motion"}
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Stage4（MoE 冻结专家 + 可训练路由）奖励项配置。"""

    # ----------------------------
    # 任务主奖励（命中/风格/多样性）
    # ----------------------------
    # effector_target_hit：
    # - 含义：每个 episode 首次有效命中给一个脉冲奖励
    # - 原始函数输出范围：{0, 1}
    # - 由于 RewardManager 内部会再乘以 dt，单次事件实际增益≈weight*dt
    # - 当前 step_dt = sim.dt * decimation = 0.005 * 4 = 0.02
    #   所以 weight=250 表示单次命中约 +250 * 0.02 = +5.0
    effector_target_hit = RewTerm(
        func=mdp.effector_target_hit,
        weight=3000.0,
        params={"command_name": "motion"},
    )
    # amp_style_reward：
    # - 含义：AMP 判别器风格奖励，鼓励动作分布接近参考动作流形
    # - 原始函数输出范围：[0, +∞)；常见训练早期在 [0, 5]，偶发更高
    # - 该项过大易压制任务目标，因此默认较小权重
    amp_style_reward = RewTerm(
        func=mdp.amp_style_reward,
        weight=0.50,
        params={"command_name": "motion"},
    )
    # router_diversity：
    # - 含义：跨 env 的群体多样性惩罚项。当某一技能在所有 env 中的平均权重
    #         过于集中（群体熵 < min_population_entropy）时施加负奖励。
    #         单个 env 内专家权重可以集中（某 env 专门用拳击或踢腿都 OK），
    #         只要不同 env 整体上分散到不同技能即可。
    # - 函数输出范围：(-min_population_entropy, 0]，全为非正值（纯惩罚项）
    # - 参数：
    #   min_population_entropy：群体熵阈值（归一化 0~1），低于此值开始惩罚
    #   参考：7 个技能均匀分布对应熵=1.0，仅 2 个技能均等对应熵≈0.36
    router_diversity = RewTerm(
        func=mdp.router_diversity_reward,
        weight=8.0,
        params={"command_name": "motion", "min_population_entropy": 0.60},
    )

    # ----------------------------
    # 命中后恢复奖励
    # ----------------------------
    # post_hit_return_to_start：
    # - 含义：命中后鼓励机器人回到初始攻击位，避免命中后直接散架
    # - 原始函数输出范围：[0, 1]（仅命中后激活）
    post_hit_return_to_start = RewTerm(
        func=mdp.post_hit_return_to_start_exp,
        weight=3.0,
        params={"command_name": "motion", "std_xy": 0.2},
    )

    # ----------------------------
    # 稳定性与正则项
    # ----------------------------
    # alive_bonus：
    # - 含义：每步存活基础正奖励，鼓励“活得久”而非冒险速死
    # - 原始函数输出范围：恒为 1
    # - 近似每秒贡献：≈ weight
    alive_bonus = RewTerm(func=mdp.alive_bonus, weight=10.0, params={"command_name": "motion"})
    # posture_unstable：
    # - 含义：躯干倾斜惩罚（带死区），限制大幅失衡但保留踢腿所需倾斜
    # - 原始函数输出范围：[-1, 0]
    # - 参数：
    #   tilt_threshold：不惩罚阈值（cos 值）
    #     当前设为 cos(20°)=0.9397，即与竖直方向夹角达到 20° 开始惩罚
    #   full_penalty_tilt：达到满惩罚的 cos 值
    #   penalty_exponent：惩罚曲线指数
    posture_unstable = RewTerm(
        func=mdp.posture_unstable,
        weight=600.0,
        params={
            "command_name": "motion",
            "tilt_threshold": 0.9397,
            "full_penalty_tilt": 0.50,
            "penalty_exponent": 2.0,
        },
    )
    # root_roll_pitch_rate：
    # - 含义：抑制躯干 roll/pitch 角速度尖峰，减少突然摔倒
    # - 原始函数输出范围：(-∞, 0]，实际常见约在 [~ -4, 0]
    # - 参数：deadband（死区，低于该角速度不惩罚）
    root_roll_pitch_rate = RewTerm(
        func=mdp.root_roll_pitch_rate_l2,
        weight=2.00,
        params={"command_name": "motion", "deadband": 1.5},
    )
    # robot_falling_penalty：
    # - 含义：机器人触发摔倒终止时的一次性惩罚
    # - 特点：该事件奖励已做 dt 归一化，实际单次惩罚≈weight（与 dt 无关）
    robot_falling_penalty = RewTerm(
        func=mdp.robot_falling_penalty_event,
        weight=-5.0,
        params={"termination_term_name": "robot_falling"},
    )
    # action_rate_l2：
    # - 含义：动作变化率惩罚，抑制高频抖动
    # - 原始函数输出范围：(-∞, 0]
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.08)
    # joint_limit：
    # - 含义：关节越界惩罚，避免不合理关节姿态
    # - 原始函数输出范围：(-∞, 0]
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # undesired_contacts：
    # - 含义：非攻击末端（手脚末端外）碰撞惩罚，降低“扑地命中”
    # - 原始函数输出范围：(-∞, 0]
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for stage-2 training."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_completed = DoneTerm(func=mdp.motion_completed, params={"command_name": "motion"})
    robot_falling = DoneTerm(
        func=mdp.robot_falling,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "height_threshold": 0.25,
            "tilt_threshold": 0.57,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        # Fallback timeout; real episode boundary is `motion_completed`.
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
