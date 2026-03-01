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
        # 新增任务导向观测 (Stage 1: Mimic 训练)
        # 使用 dummy/关联数据，为 Stage 2 任务导向 RL 做准备
        # =====================================================================
        
        # (1) 目标相对位置: 3 维
        # Stage 1: 使用参考动作中攻击肢体的位置作为 dummy target
        target_rel_pos = ObsTerm(
            func=mdp.target_relative_position,
            params={"command_name": "motion", "effector_body_name": "right_ankle_roll_link"},
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
        # 新增任务导向观测 (Stage 1: Mimic 训练)
        # Critic 也需要这些观测来准确估计价值函数
        # =====================================================================
        
        # (1) 目标相对位置: 3 维
        target_rel_pos = ObsTerm(
            func=mdp.target_relative_position,
            params={"command_name": "motion", "effector_body_name": "right_ankle_roll_link"}
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
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    
    # =========================================================================
    # 新增: 攻击肢体靠近目标位置的奖励 (Stage 1 辅助)
    # 权重设置较小 (0.2)，不影响 mimic 主导地位
    # Stage 2 时可以增大权重
    # =========================================================================
    effector_target = RewTerm(
        func=mdp.effector_target_tracking_exp,
        weight=0.0,
        params={
            "command_name": "motion",
            "std": 0.25,  # 较小的 std 让奖励对距离更敏感
            "effector_body_name": "right_ankle_roll_link",
        },
    )
    
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
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
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
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
