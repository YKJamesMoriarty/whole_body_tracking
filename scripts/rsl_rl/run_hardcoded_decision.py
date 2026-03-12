"""Stage 3 硬编码决策层实验脚本.

实验流程:
1. 从 label 目录读取离线打标结果，并按命中率阈值筛出可用目标点
2. 将可用点按技能分簇（默认 cross/swing/roundhouse）
3. 按固定循环执行连续 trial（非每轮 reset）:
   stance(目标在地下) -> 等待 stance_duration_s -> 切换攻击技能并设置目标 -> hit/超时/终止

说明:
- 脚本不会在每个 trial 开头主动 reset 环境，避免“硬切换”。
- 仅当环境返回 done（例如摔倒）时，才执行 reset 恢复继续。
"""

# ====================================================================
# 第一步: Isaac Sim 初始化 (必须最先执行)
# ====================================================================
import argparse
import json
import pathlib
import sys
from collections import Counter
from collections.abc import Callable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Stage 3 硬编码决策层实验")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Isaac Lab 任务名")
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数。当前脚本按单机器人流程设计，建议=1")
parser.add_argument("--num_episodes", type=int, default=40, help="总 episode 数")
parser.add_argument("--num_trials", type=int, default=None, help="总连续切换回合数（若设置则覆盖 --num_episodes）")
parser.add_argument(
    "--decision_cycle",
    type=str,
    nargs="+",
    default=["cross_right", "swing_right", "roundhouse_right"],
    help="技能循环顺序（新7技能名：cross_right/swing_right/hook_left/roundhouse_right/roundhouse_left/frontkick_right）",
)
parser.add_argument(
    "--use_roundhouse_fixed_points",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="是否对 roundhouse_right/roundhouse_left 使用固定目标点列表采样（用于 demo 稳定性）",
)
parser.add_argument(
    "--roundhouse_fixed_points",
    type=str,
    default="0.8,0.0,-0.05",
    help="roundhouse 固定目标点列表，格式: x,y,z;x,y,z",
)
parser.add_argument("--labels_dir", type=str, default="label", help="label_skills.py 输出目录")
parser.add_argument("--min_accuracy", type=float, default=0.8, help="可用目标点的最小命中率阈值")
parser.add_argument("--min_target_distance", type=float, default=0.3, help="目标点最小距离阈值（米）")
parser.add_argument("--max_target_distance", type=float, default=0.9, help="目标点最大距离阈值（米）")
parser.add_argument("--stance_duration_s", type=float, default=2.0, help="每回合 stance 持续时间（秒）")
parser.add_argument("--attack_timeout_s", type=float, default=8.0, help="每回合攻击超时（秒）")
parser.add_argument("--post_hit_recovery_s", type=float, default=1.0, help="命中后恢复时间（秒），结束后再切 stance")
parser.add_argument("--model_dir", type=str, default="basic_model/Mimic", help=".pt 模型目录")
parser.add_argument("--motion_dir", type=str, default="iros_motion/npz", help=".npz 动作目录")
parser.add_argument("--output_dir", type=str, default="outputs/stage3_decision", help="实验记录输出目录")
parser.add_argument("--seed", type=int, default=0, help="随机种子")
parser.add_argument(
    "--avoid_motion_resample",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="在 trial 内避免 MotionCommand 到尾帧触发硬重采样（提升连续观感）",
)
parser.add_argument(
    "--debug_vis",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="是否开启命令可视化（目标球等）",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ====================================================================
# 第二步: 正式 import（Isaac Sim 已启动）
# ====================================================================
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401

from whole_body_tracking.tasks.tracking.stage3.skill_config import SKILL_CONFIGS, get_model_path, get_motion_path
from whole_body_tracking.tasks.tracking.stage3.target_bank import LabeledTargetBank


def _get_command(env: RslRlVecEnvWrapper):
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

    return env.unwrapped.command_manager.get_term("motion")  # type: ignore[return-value]


def _load_policy(model_path: pathlib.Path, env: RslRlVecEnvWrapper, device: str):
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

    agent_cfg = G1FlatPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(model_path))
    return runner.get_inference_policy(device=device)


def _reset_env_get_obs(env: RslRlVecEnvWrapper):
    out = env.reset()
    if isinstance(out, tuple):
        obs = out[0]
    else:
        obs = out
    if obs is None:
        obs = env.get_observations()
    return obs


def _is_env0_done(dones: torch.Tensor | np.ndarray | list) -> bool:
    if isinstance(dones, torch.Tensor):
        return bool(dones[0].item())
    if isinstance(dones, np.ndarray):
        return bool(dones[0])
    return bool(dones[0])


def _switch_skill(command, skill_name: str, motion_base: pathlib.Path):
    cfg = SKILL_CONFIGS[skill_name]
    command.reload_motion(str(get_motion_path(skill_name, motion_base)))
    command.switch_skill(
        effector_body_name=cfg["effector_body_name"],
        skill_type_idx=cfg["skill_type_idx"],
        effector_one_hot_idx=cfg["effector_one_hot_idx"],
    )
    # 切技能时从该技能动作首帧开始，避免继承上一技能 time_steps 导致瞬时越界重采样
    command.time_steps[:] = 0
    command.current_time[:] = 0.0


def _rewind_motion_if_needed(command, reserve_steps: int = 2):
    """避免 time_steps 触发 _resample_command 的硬重置。"""
    max_step = int(command.motion.time_step_total) - max(1, reserve_steps)
    if max_step <= 0:
        return
    near_end = command.time_steps >= max_step
    if torch.any(near_end):
        command.time_steps[near_end] = 0


def _parse_point_list(points_text: str) -> list[np.ndarray]:
    """解析 'x,y,z;x,y,z' 为点列表。"""
    text = points_text.strip()
    if len(text) == 0:
        return []
    points: list[np.ndarray] = []
    for token in text.split(";"):
        parts = [p.strip() for p in token.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"无效点格式: '{token}'，应为 x,y,z（多个点用分号分隔）"
            )
        point = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
        points.append(point)
    return points


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):  # type: ignore[override]
    if args_cli.num_envs != 1:
        raise ValueError("当前脚本按单机器人串行逻辑实现，请使用 --num_envs 1")
    if args_cli.min_target_distance < 0.0 or args_cli.max_target_distance < 0.0:
        raise ValueError("距离阈值必须 >= 0")
    if args_cli.post_hit_recovery_s < 0.0:
        raise ValueError("post_hit_recovery_s 必须 >= 0")
    if args_cli.min_target_distance > args_cli.max_target_distance:
        raise ValueError(
            f"min_target_distance ({args_cli.min_target_distance}) 不能大于 "
            f"max_target_distance ({args_cli.max_target_distance})"
        )

    num_trials = args_cli.num_trials if args_cli.num_trials is not None else args_cli.num_episodes
    roundhouse_fixed_points = _parse_point_list(args_cli.roundhouse_fixed_points)
    if args_cli.use_roundhouse_fixed_points and len(roundhouse_fixed_points) == 0:
        raise ValueError("已启用 roundhouse 固定点，但 --roundhouse_fixed_points 为空")

    project_root = pathlib.Path(__file__).resolve().parents[2]
    model_base = project_root / args_cli.model_dir
    motion_base = project_root / args_cli.motion_dir
    output_dir = project_root / args_cli.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    decision_cycle = list(args_cli.decision_cycle)
    for skill_name in decision_cycle + ["stance"]:
        if skill_name not in SKILL_CONFIGS:
            raise ValueError(f"未知技能: {skill_name}, 可用: {list(SKILL_CONFIGS.keys())}")

    target_bank = LabeledTargetBank.from_dir(
        labels_dir=project_root / args_cli.labels_dir,
        attack_skills=decision_cycle,
        min_accuracy=args_cli.min_accuracy,
        min_distance=args_cli.min_target_distance,
        max_distance=args_cli.max_target_distance,
    )
    cluster_sizes = target_bank.summary()
    for skill_name, size in cluster_sizes.items():
        if size == 0:
            raise RuntimeError(
                f"技能 {skill_name} 在当前过滤条件下无可用点："
                f"min_accuracy={args_cli.min_accuracy:.3f}, "
                f"distance=[{args_cli.min_target_distance:.3f}, {args_cli.max_target_distance:.3f}]。"
                "请调小阈值。"
            )

    print("\n" + "=" * 72)
    print("[Stage3] 硬编码决策层实验")
    print(f"  task={args_cli.task} | trials={num_trials} | cycle={decision_cycle}")
    print(
        f"  stance={args_cli.stance_duration_s:.2f}s | attack_timeout={args_cli.attack_timeout_s:.2f}s"
    )
    print(f"  post_hit_recovery={args_cli.post_hit_recovery_s:.2f}s")
    print(f"  labels={args_cli.labels_dir} | min_accuracy={args_cli.min_accuracy:.2f}")
    print(
        f"  target_distance_range=[{args_cli.min_target_distance:.2f}, {args_cli.max_target_distance:.2f}] m"
    )
    print(
        f"  roundhouse_fixed_points={'on' if args_cli.use_roundhouse_fixed_points else 'off'} | "
        f"points={roundhouse_fixed_points}"
    )
    print(f"  avoid_motion_resample={args_cli.avoid_motion_resample}")
    print(f"  candidate_points={cluster_sizes}")
    print("=" * 72 + "\n")

    # 用 stance 初始化环境，随后在脚本中动态切换技能
    stance_cfg = SKILL_CONFIGS["stance"]
    env_cfg.scene.num_envs = 1
    env_cfg.commands.motion.motion_file = str(get_motion_path("stance", motion_base))
    env_cfg.commands.motion.effector_body_name = stance_cfg["effector_body_name"]
    env_cfg.commands.motion.skill_type_idx = stance_cfg["skill_type_idx"]
    env_cfg.commands.motion.effector_one_hot_idx = stance_cfg["effector_one_hot_idx"]
    env_cfg.commands.motion.debug_vis = args_cli.debug_vis
    env_cfg.commands.motion.spawn_target_delay = 0.0
    env_cfg.commands.motion.hit_resample_delay = max(30.0, args_cli.attack_timeout_s + 5.0)
    # 设为足够长，避免 time_out 在 demo 中频繁打断连续技能切换
    env_cfg.episode_length_s = max(
        3600.0,
        num_trials * (args_cli.stance_duration_s + args_cli.attack_timeout_s + 1.0) + 60.0,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device
    command = _get_command(env)

    # 只加载本次实验会用到的策略
    needed_skills = ["stance"] + [s for s in decision_cycle if s != "stance"]
    policies: dict[str, Callable] = {}
    for skill_name in needed_skills:
        model_path = get_model_path(skill_name, model_base)
        print(f"[Stage3] 加载策略: {skill_name:10s} <- {model_path.name}")
        policies[skill_name] = _load_policy(model_path, env, device)

    sim_dt = float(env_cfg.decimation * env_cfg.sim.dt)
    stance_steps = max(1, int(np.ceil(args_cli.stance_duration_s / sim_dt)))
    attack_steps_max = max(1, int(np.ceil(args_cli.attack_timeout_s / sim_dt)))
    post_hit_recovery_steps = int(np.ceil(args_cli.post_hit_recovery_s / sim_dt))
    rng = np.random.default_rng(args_cli.seed)

    underground_target = torch.tensor([[0.0, 0.0, -10.0]], device=device, dtype=torch.float32)
    env0 = torch.tensor([0], device=device, dtype=torch.long)

    episode_records = []
    outcome_counter: Counter[str] = Counter()
    reset_counter = 0
    obs = _reset_env_get_obs(env)

    for trial_idx in range(num_trials):
        attack_skill = decision_cycle[trial_idx % len(decision_cycle)]
        if attack_skill in ("roundhouse_right", "roundhouse_left") and args_cli.use_roundhouse_fixed_points:
            pick_idx = int(rng.integers(0, len(roundhouse_fixed_points)))
            target_local_np = roundhouse_fixed_points[pick_idx].copy()
            target_source = "roundhouse_fixed"
        else:
            target_local_np = target_bank.sample_target(attack_skill, rng)
            target_source = "label_cluster"
        target_local = torch.tensor(target_local_np[None, :], device=device, dtype=torch.float32)

        # Phase A: 站姿准备（目标在地下）
        _switch_skill(command, "stance", motion_base)
        command.set_per_env_targets(env0, underground_target)
        for _ in range(stance_steps):
            if args_cli.avoid_motion_resample:
                _rewind_motion_if_needed(command)
            with torch.inference_mode():
                actions = policies["stance"](obs)
            obs, _, dones, _ = env.step(actions)
            if _is_env0_done(dones):
                obs = _reset_env_get_obs(env)
                reset_counter += 1

        # Phase B: 切换攻击技能并执行，直到 hit / timeout / terminated
        _switch_skill(command, attack_skill, motion_base)
        command.set_per_env_targets(env0, target_local)
        command.cumulative_hit_count[:] = 0.0

        outcome = "timeout"
        attack_steps_used = 0
        recovery_steps_used = 0
        for step_i in range(attack_steps_max):
            if args_cli.avoid_motion_resample:
                _rewind_motion_if_needed(command)
            with torch.inference_mode():
                actions = policies[attack_skill](obs)
            obs, _, dones, _ = env.step(actions)
            attack_steps_used = step_i + 1

            if command.cumulative_hit_count[0].item() > 0:
                outcome = "hit"
                break
            if _is_env0_done(dones):
                outcome = "terminated"
                obs = _reset_env_get_obs(env)
                reset_counter += 1
                break

        # 命中后给出恢复窗口，避免立刻切 stance 带来生硬切换
        if outcome == "hit" and post_hit_recovery_steps > 0:
            for _ in range(post_hit_recovery_steps):
                if args_cli.avoid_motion_resample:
                    _rewind_motion_if_needed(command)
                with torch.inference_mode():
                    actions = policies[attack_skill](obs)
                obs, _, dones, _ = env.step(actions)
                recovery_steps_used += 1
                if _is_env0_done(dones):
                    outcome = "hit_then_terminated"
                    obs = _reset_env_get_obs(env)
                    reset_counter += 1
                    break

        outcome_counter[outcome] += 1
        attack_time_s = attack_steps_used * sim_dt
        recovery_time_s = recovery_steps_used * sim_dt
        record = {
            "trial": trial_idx + 1,
            "skill": attack_skill,
            "target_source": target_source,
            "target_local_xyz": [float(target_local_np[0]), float(target_local_np[1]), float(target_local_np[2])],
            "outcome": outcome,
            "attack_time_s": float(attack_time_s),
            "recovery_time_s": float(recovery_time_s),
        }
        episode_records.append(record)

        print(
            f"[TRIAL {trial_idx+1:04d}] skill={attack_skill:10s} "
            f"src={target_source:16s} "
            f"target=({target_local_np[0]:+.3f},{target_local_np[1]:+.3f},{target_local_np[2]:+.3f}) "
            f"outcome={outcome:18s} attack_t={attack_time_s:.2f}s recovery_t={recovery_time_s:.2f}s"
        )

    summary = {
        "num_trials": num_trials,
        "decision_cycle": decision_cycle,
        "min_accuracy": float(args_cli.min_accuracy),
        "min_target_distance": float(args_cli.min_target_distance),
        "max_target_distance": float(args_cli.max_target_distance),
        "use_roundhouse_fixed_points": bool(args_cli.use_roundhouse_fixed_points),
        "roundhouse_fixed_points": [p.tolist() for p in roundhouse_fixed_points],
        "stance_duration_s": float(args_cli.stance_duration_s),
        "attack_timeout_s": float(args_cli.attack_timeout_s),
        "post_hit_recovery_s": float(args_cli.post_hit_recovery_s),
        "continuous_mode": True,
        "avoid_motion_resample": bool(args_cli.avoid_motion_resample),
        "env_episode_length_s": float(env_cfg.episode_length_s),
        "env_reset_count": int(reset_counter),
        "cluster_sizes": cluster_sizes,
        "outcome_counts": dict(outcome_counter),
        "hit_rate": float(outcome_counter.get("hit", 0) / max(1, num_trials)),
        "records": episode_records,
    }

    out_path = output_dir / "hardcoded_decision_log.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"[Stage3] 完成，日志已保存: {out_path}")
    print(f"[Stage3] outcome 统计: {dict(outcome_counter)} | hit_rate={summary['hit_rate']:.3f}")
    print("=" * 72 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
