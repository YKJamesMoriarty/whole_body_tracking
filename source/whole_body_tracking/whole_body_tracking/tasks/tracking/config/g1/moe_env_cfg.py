from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs.mdp.rewards import is_alive
from whole_body_tracking.tasks.tracking.moe import events as moe_events

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    EventCfg,
    MySceneCfg,
    TrackingEnvCfg,
)
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.moe import actions as moe_actions
from whole_body_tracking.tasks.tracking.moe import commands as moe_commands
from whole_body_tracking.tasks.tracking.moe import observations as moe_observations
from whole_body_tracking.tasks.tracking.moe import rewards as moe_rewards
from whole_body_tracking.tasks.tracking.moe import terminations as moe_terminations


@configclass
class MoECommandsCfg:
    attack_target = moe_commands.AttackTargetCommandCfg(
        target_file="outputs/attack_target_final/moe_targets.npz",
        visible_steps=None,
        debug_vis=True,
    )


@configclass
class MoEObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        target_pos_b = ObsTerm(func=moe_observations.target_pos_b)
        target_present = ObsTerm(func=moe_observations.target_present)
        time_since_switch = ObsTerm(func=moe_observations.time_since_switch)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        target_pos_b = ObsTerm(func=moe_observations.target_pos_b)
        target_present = ObsTerm(func=moe_observations.target_present)
        time_since_switch = ObsTerm(func=moe_observations.time_since_switch)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class MoERewardsCfg:
    hit = RewTerm(func=moe_rewards.target_hit, weight=5.0, params={"hit_radius": 0.12})
    progress = RewTerm(func=moe_rewards.target_progress, weight=3.0, params={"clamp": 0.05})
    alive = RewTerm(func=is_alive, weight=0.5)
    entropy = RewTerm(func=moe_rewards.moe_weight_entropy, weight=-0.02)
    skill_match = RewTerm(func=moe_rewards.skill_match, weight=1.0)
    switch_penalty = RewTerm(func=moe_rewards.switch_penalty, weight=-2.00)


@configclass
class MoETerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen, params={"minimum_height": 0.25, "limit_angle": 1.0472})
    hit_target = DoneTerm(func=moe_terminations.hit_target, params={"hit_radius": 0.12})


@configclass
class G1FlatMoEEnvCfg(TrackingEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=1024, env_spacing=2.5)
    observations: MoEObservationsCfg = MoEObservationsCfg()
    actions: moe_actions.MoEActionsCfg = moe_actions.MoEActionsCfg()
    commands: MoECommandsCfg = MoECommandsCfg()
    rewards: MoERewardsCfg = MoERewardsCfg()
    terminations: MoETerminationsCfg = MoETerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.moe.scale = G1_ACTION_SCALE
        self.actions.moe.model_dir = "basic_model/Mimic_refine"
        self.actions.moe.motion_dir = "iros_motion/npz"
        self.actions.moe.joint_names = [".*"]
        self.actions.moe.hard_gate = True
        self.actions.moe.lock_skill_per_episode = True
        self.actions.moe.anchor_body_name = "torso_link"
        self.actions.moe.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        # Disable joint default position randomization since MoE uses a custom action term (no "joint_pos").
        self.events.add_joint_default_pos = None
        # Reset robot pose each episode to a stable stance frame.
        self.events.reset_scene = EventTerm(
            func=moe_events.reset_to_stance_motion,
            mode="reset",
            params={},
        )
        # Visualize target hit radius (when rendering).
        self.commands.attack_target.visual_radius = self.rewards.hit.params.get("hit_radius", 0.12)
        self.commands.attack_target.hit_rate_threshold = 0.95
        self.commands.attack_target.auto_stage_switch = True

        # shorter episodes for single skill execution
        self.episode_length_s = 5.0
