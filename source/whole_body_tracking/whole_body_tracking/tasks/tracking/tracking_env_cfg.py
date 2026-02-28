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
    # =========================================================================
    # Stage 2  强化末端执行器（右手）的奖励
    # =========================================================================
    # Mimic 右手末端位置奖励
    # mimic_right_hand_pos = RewTerm(
    #     func=mdp.mimic_right_hand_position_exp,
    #     weight=0.0,  # 可根据实验调整
    #     params={"command_name": "motion", "std": 0.3},
    # )
    # Mimic 右手末端旋转奖励
    # mimic_right_hand_ori = RewTerm(
    #     func=mdp.mimic_right_hand_orientation_exp,
    #     weight=0.0,  # 可根据实验调整
    #     params={"command_name": "motion", "std": 0.3},
    # )
    
    # Mimic 右肘关节 DOF 奖励
    mimic_right_elbow_dof = RewTerm(
        func=mdp.mimic_right_elbow_dof_exp,
        weight=1.0,  # 可根据实验调整
        params={"command_name": "motion", "std": 0.3},
    )
    # Mimic 右肩外展关节 DOF 奖励
    mimic_right_shoulder_roll_dof = RewTerm(
        func=mdp.mimic_right_shoulder_roll_dof_exp,
        weight=1.0,  # 可根据实验调整
        params={"command_name": "motion", "std": 0.3},
    )
    """Reward terms for the MDP.
    
    Stage 2: Task-Oriented RL 奖励配置 (进展奖励版)
    ===================================
    
    奖励设计原则:
    1. 核心任务奖励 (effector_target_hit) 权重最高，明确训练目标
    2. 进展奖励 (effector_target_near) 解决稀疏奖励探索问题
    3. Mimic 奖励权重保留，维持动作质量
    4. 进展奖励机制自然防止蹭分，无需额外惩罚
    5. Hit 后 1s 延迟重采样，鼓励收手跟随参考动作
    """
    
    # =========================================================================
    # Stage 2 核心任务奖励
    # =========================================================================
    
    # [核心] 有效击中目标 - 最重要的奖励信号
    # 权重 20.0: 脉冲式奖励，Hit 后目标会在 1s 后重采样
    effector_target_hit = RewTerm(
        func=mdp.effector_target_hit,
        weight=12.0,
        params={"command_name": "motion"},
    )
    
    # [引导] 进展奖励 - 只有比历史最近距离更近时才给奖励
    # 设计优点:
    # - 手停在原地: 没有奖励 (距离没变近)
    # - 手绕圈: 没有奖励 (距离没变近)
    # - 手向目标移动: 有奖励 (与接近量成正比)
    # - 手 Hit 到目标: 之后 Near 奖励自然归零 (不可能比 0 更近)
    # 
    # 权重设计:
    # - Mimic 奖励约 0.5~0.9 * 6项 ≈ 3~4 每 step
    # - Near 累计奖励 = 0.25m * 10.0 * 3.0 = 7.5 (整个进攻过程)
    # - 设为 3.0 让 Near 奖励有足够吸引力，鼓励机器人主动进攻
    # 权重最好不要加到30以上，那样的话最后机器人会只那near和hit，然后就摔倒；
    effector_target_near = RewTerm(
        func=mdp.effector_target_near,
        weight=15.0,
        params={
            "command_name": "motion",
            "guidance_radius": 0.4,  # 引导球半径
            "scale": 10.0,  # 每接近 1cm 奖励 0.1
        },
    )
    
    # [战术] 躯干朝向目标 - 鼓励正确的攻击姿态
    effector_face_target = RewTerm(
        func=mdp.effector_face_target,
        weight=1.0,
        params={"command_name": "motion"},
    )

    effector_velocity_towards_target = RewTerm(
        func=mdp.effector_velocity_towards_target,
        weight=1.0,  # 可调整
        params={
            "command_name": "motion",
            "guidance_radius": 0.4,
        },
    )
    
    # =========================================================================
    # Stage 2 收手阶段奖励/惩罚 (Hit 后 0.2~1.0s 冷静期内)
    # 
    # 设计原则:
    # - 只在冷静期内生效，鼓励机器人收手
    # - 与进攻阶段的 Near 奖励对称设计
    # - 配合 Mimic 奖励，引导跟随参考动作
    # =========================================================================
    
    # [收手惩罚-小球] 手停留在目标小球内的惩罚
    # Hit 后 0.2s 开始，如果手还在小球内就惩罚
    # pen_linger_in_hit_sphere = RewTerm(
    #     func=mdp.pen_linger_in_hit_sphere,
    #     weight=0.0,  # 权重为正，函数返回负值
    #     params={
    #         "command_name": "motion",
    #         "grace_period": 0.2,
    #     },
    # )
    
    # [收手奖励-大球] 手离开目标的进展奖励 (与 Near 对称)
    # 权重与 Near 相同，形成对称的进攻-收手周期
    # rew_retract_from_target = RewTerm(
    #     func=mdp.rew_retract_from_target,
    #     weight=0.0,  # 与 effector_target_near 相同
    #     params={
    #         "command_name": "motion",
    #         "guidance_radius": 0.4,
    #         "grace_period": 0.2,
    #         "scale": 10.0,  # 与 Near 相同
    #     },
    # )
    
    # =========================================================================
    # Stage 2 惩罚项 (姿态稳定性)
    # =========================================================================
    
    # [姿态] 惩罚身体过度倾斜
    posture_unstable = RewTerm(
        func=mdp.posture_unstable,
        weight=2000.0,  # 权重为正，函数返回负值
        params={
            "command_name": "motion",
            "tilt_threshold": 0.0097,  # 约8度
        },
    )
    
    # =========================================================================
    # Stage 2 调整后的 Mimic 奖励 (保留小权重，维持动作质量)
    # 原来stage1中以下6个奖励的权重分别为0.5，0.5，1，1，1，1
    # 设计原因:
    # - 保留 0.1 的小权重而不是完全设为 0
    # - 防止长时间 Stage 2 训练后出现多余的移动或奇怪姿态
    # - cross 数据是原地出拳，不需要大幅移动，所以保留位置约束有意义
    # =========================================================================

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=5.0,  # Stage 2: 跟踪 anchor (torso) 位置，保持机器人不漫游
        params={"command_name": "motion", "std": 0.25},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,  # Stage 2: 保留朝向约束
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=5.0,  # Stage 2: 跟踪全身 14 个 body 位置，保持出拳姿态 (必须 > 0!)
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=5.0,  # Stage 2: 保持关节朝向 (必须 > 0!)
        params={"command_name": "motion", "std": 0.5},
    )
    # 除去右手以外其他11个身体部分link的mimic
    mimic_non_right_hand_body_pos = RewTerm(
        func=mdp.mimic_non_right_hand_body_position_error_exp,
        weight=2.0,  # 可根据实验调整
        params={"command_name": "motion", "std": 0.3},
    )
    # Mimic 非右手身体部分姿态奖励
    mimic_non_right_hand_body_ori = RewTerm(
        func=mdp.mimic_non_right_hand_body_orientation_error_exp,
        weight=2.0,  # 可根据实验调整
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,  # Stage 2: 保留速度约束
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,  # Stage 2: 保留角速度约束
        params={"command_name": "motion", "std": 3.14},
    )
    
    # =========================================================================
    # 保持不变的正则化惩罚
    # =========================================================================
    # 某些时候会出现极大负数值。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    
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
    """Termination terms for the MDP.
    
    Stage 2 设计说明:
    ================
    Stage 1 (Mimic) 的终止条件基于参考动作的跟踪误差:
    - anchor_pos: 锚点位置偏差过大
    - anchor_ori: 锚点朝向偏差过大  
    - ee_body_pos: 四肢位置偏差过大
    
    Stage 2 (Task-Oriented) 不再强约束跟踪参考动作:
    - 删除上述 mimic 相关的终止条件
    - 添加物理稳定性检测 (摔倒检测)
    - 保留 time_out 防止无限循环
    """

    # 正常超时终止 - 保留
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Stage 2: 摔倒检测 - 替代 mimic 相关的终止条件
    # 条件: 骨盆高度 < 0.3m 或 躯干倾斜角 > ~55度 (cos < 0.57)
    robot_falling = DoneTerm(
        func=mdp.robot_falling,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "height_threshold": 0.25,   # 骨盆最低高度 (米)
            "tilt_threshold": 0.57,    # 倾斜余弦阈值 (约55度，cos(55°)≈0.574)
        },
    )
    
    # [已删除] Stage 1 Mimic 终止条件 - Stage 2 不需要
    # anchor_pos = DoneTerm(...)  # 锚点位置偏差
    # anchor_ori = DoneTerm(...)  # 锚点朝向偏差
    # ee_body_pos = DoneTerm(...) # 四肢位置偏差


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
        self.episode_length_s = 20.0  # Stage 2: 增加到 20 秒，给机器人更多时间尝试击打目标
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
