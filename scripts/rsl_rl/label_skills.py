"""Stage 3 离线仿真打标脚本 (Offline Simulation Labeling)

对采样空间内的目标点，依次测试各进攻技能的命中率，
生成供高层分类器训练用的标签数据。

采样空间 (机器人局部坐标系, Pelvis 为原点):
    x: 0.0 ~ 1.2 m  (前方)
    y: -0.5 ~ 0.5 m (左右)
    z: -0.4 ~ 0.4 m (高低)

使用方法:
    python scripts/rsl_rl/label_skills.py \\
        --task Tracking-Flat-G1-v0 \\
        --num_envs 512 \\
        --grid_spacing 0.05 \\
        --n_episodes 10 \\
        --model_dir basic_model \\
        --motion_dir Reference_Motion_IROS \\
        --output_dir labels

输出文件 (保存到 output_dir/):
    grid_points.npy      : (N, 3) 采样点局部坐标
    all_accuracies.npy   : (N, K) 各技能命中率矩阵
    labels.npy           : (N,)   最优技能 ID (无命中时为 stance_id=5)
    metadata.json        : 采样参数和技能 ID 映射

Isaac Sim 必须在最开始初始化，因此参数解析在 import 之前完成。
"""

# ====================================================================
# 第一步: Isaac Sim 初始化 (必须最先执行)
# ====================================================================
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 3 离线仿真打标脚本")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Isaac Lab 任务名")
parser.add_argument("--num_envs", type=int, default=512, help="并行环境数量")
parser.add_argument(
    "--grid_spacing", type=float, default=0.05,
    help="目标采样间隔 (m)，推荐 0.05（~8925 点）或 0.10（~1287 点，更快）"
)
parser.add_argument("--n_episodes", type=int, default=10, help="每个 (target, skill) 组合测试次数")
parser.add_argument(
    "--skills", type=str, nargs="+",
    default=["cross", "swing", "roundhouse", "frontkick"],
    help="要测试的技能名称（顺序决定测试顺序）"
)
parser.add_argument("--model_dir", type=str, default="basic_model", help=".pt 文件所在目录（相对项目根目录）")
parser.add_argument("--motion_dir", type=str, default="Reference_Motion_IROS", help=".npz 文件所在目录")
parser.add_argument("--output_dir", type=str, default="labels", help="结果输出目录")
parser.add_argument(
    "--count_fall_after_hit_as_success",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "是否将“先命中后摔倒”的 episode 记为命中成功。"
        "True=沿用旧逻辑；False=仅命中且未摔倒才算成功。"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ====================================================================
# 第二步: 正式 import（Isaac Sim 已启动）
# ====================================================================
import json
import pathlib
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401 触发任务注册

from whole_body_tracking.tasks.tracking.stage3.skill_config import (
    SKILL_CONFIGS,
    STANCE_SKILL_ID,
    get_model_path,
    get_motion_path,
)


# ====================================================================
# 辅助函数
# ====================================================================

def _get_command(env: RslRlVecEnvWrapper):
    """从 VecEnv 包装中取出 MotionCommand 对象"""
    from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
    return env.unwrapped.command_manager.get_term("motion")  # type: ignore[return-value]


def _load_policy(model_path: str | pathlib.Path, env: RslRlVecEnvWrapper, device: str):
    """从本地 .pt 文件加载 RSL-RL 推理策略"""
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    agent_cfg = G1FlatPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(model_path))
    return runner.get_inference_policy(device=device)


def _make_grid(spacing: float) -> np.ndarray:
    """生成局部坐标系下的目标采样网格 (N, 3)"""
    half = spacing / 2  # 确保端点被包含
    x_vals = np.arange(0.0,  1.2  + half, spacing)
    y_vals = np.arange(-0.5, 0.5  + half, spacing)
    z_vals = np.arange(-0.4, 0.4  + half, spacing)
    grid = np.array([[x, y, z] for x in x_vals for y in y_vals for z in z_vals], dtype=np.float32)
    return grid


# ====================================================================
# 主函数
# ====================================================================

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):  # type: ignore[override]
    # ------------------------------------------------------------------
    # 1. 解析路径
    # ------------------------------------------------------------------
    project_root = pathlib.Path(__file__).resolve().parents[2]
    model_base   = project_root / args_cli.model_dir
    motion_base  = project_root / args_cli.motion_dir
    output_dir   = project_root / args_cli.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    skills_to_test: list[str] = args_cli.skills
    for s in skills_to_test:
        if s not in SKILL_CONFIGS:
            raise ValueError(f"未知技能名: '{s}'，可用: {list(SKILL_CONFIGS.keys())}")

    # ------------------------------------------------------------------
    # 2. 生成采样网格
    # ------------------------------------------------------------------
    grid_points = _make_grid(args_cli.grid_spacing)   # (N, 3)
    n_points    = len(grid_points)
    n_skills    = len(skills_to_test)
    n_episodes  = args_cli.n_episodes
    num_envs    = args_cli.num_envs

    print(f"\n{'='*60}")
    print(f"[打标] 采样间隔 {args_cli.grid_spacing}m → {n_points} 个目标点")
    print(f"[打标] 测试技能: {skills_to_test}")
    print(f"[打标] 每点每技能: {n_episodes} 次 | 并行 env: {num_envs}")
    print(f"[打标] 总测试次数: {n_points} × {n_skills} × {n_episodes} = {n_points*n_skills*n_episodes}")
    print(f"[打标] 先命中后摔倒是否算成功: {args_cli.count_fall_after_hit_as_success}")
    print(f"{'='*60}\n")

    # 提前保存网格坐标和基础参数，方便中断后用已有的 skill_accuracy.npy 恢复
    np.save(output_dir / "grid_points.npy", grid_points)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "grid_spacing":   args_cli.grid_spacing,
            "n_episodes":     n_episodes,
            "num_envs":       num_envs,
            "skills_tested":  skills_to_test,
            "skill_ids":      [SKILL_CONFIGS[s]["skill_id"] for s in skills_to_test],
            "stance_skill_id": STANCE_SKILL_ID,
            "n_points":       n_points,
            "x_range":   [0.0,  1.2],
            "y_range":   [-0.5, 0.5],
            "z_range":   [-0.4, 0.4],
            "count_fall_after_hit_as_success": args_cli.count_fall_after_hit_as_success,
            "status": "in_progress",  # 完成后会覆盖为包含 label_distribution 的完整版本
        }, f, indent=2, ensure_ascii=False)
    print(f"[打标] 网格坐标已预存至 {output_dir}/grid_points.npy")

    # ------------------------------------------------------------------
    # 3. 修改 env_cfg 以适配打标场景
    # ------------------------------------------------------------------
    env_cfg.scene.num_envs    = num_envs
    env_cfg.episode_length_s  = 8.0    # 每次进攻尝试 8s
    env_cfg.commands.motion.spawn_target_delay = 0.5  # 缩短 spawn 延迟节省时间
    if args_cli.count_fall_after_hit_as_success:
        # 旧逻辑：命中后短延迟重采样，episode 内可继续出现新目标
        env_cfg.commands.motion.hit_resample_delay = 0.5
    else:
        # 严格逻辑：首个 hit 后目标保持地下，直到 episode 结束（timeout / fall）
        env_cfg.commands.motion.hit_resample_delay = env_cfg.episode_length_s + 1.0
    env_cfg.commands.motion.debug_vis = False          # 关闭可视化提升速度

    # 使用第一个技能的配置初始化 env
    first_cfg = SKILL_CONFIGS[skills_to_test[0]]
    env_cfg.commands.motion.motion_file       = str(get_motion_path(skills_to_test[0], motion_base))
    env_cfg.commands.motion.effector_body_name = first_cfg["effector_body_name"]
    env_cfg.commands.motion.skill_type_idx     = first_cfg["skill_type_idx"]
    env_cfg.commands.motion.effector_one_hot_idx = first_cfg["effector_one_hot_idx"]

    # ------------------------------------------------------------------
    # 4. 创建仿真环境 (Isaac Sim 初始化)
    # ------------------------------------------------------------------
    print("[打标] 初始化仿真环境...")
    env     = gym.make(args_cli.task, cfg=env_cfg)
    env     = RslRlVecEnvWrapper(env)
    device  = env.unwrapped.device
    command = _get_command(env)

    # ------------------------------------------------------------------
    # 5. 结果矩阵: all_accuracies[i, j] = 目标点 i 用技能 j 的命中率
    # ------------------------------------------------------------------
    all_accuracies = np.zeros((n_points, n_skills), dtype=np.float32)

    # ------------------------------------------------------------------
    # 6. 对每个技能进行打标
    # ------------------------------------------------------------------
    for skill_idx, skill_name in enumerate(skills_to_test):
        skill_cfg  = SKILL_CONFIGS[skill_name]
        model_path = get_model_path(skill_name, model_base)
        motion_path = get_motion_path(skill_name, motion_base)

        print(f"\n[打标] ── 技能 {skill_idx+1}/{n_skills}: {skill_name} ({skill_cfg['description']}) ──")
        print(f"  模型: {model_path}")
        print(f"  动作: {motion_path}")

        # 切换技能（第一个技能在 env_cfg 中已配置，后续动态切换）
        if skill_idx > 0:
            command.reload_motion(str(motion_path))
            command.switch_skill(
                effector_body_name    = skill_cfg["effector_body_name"],
                skill_type_idx        = skill_cfg["skill_type_idx"],
                effector_one_hot_idx  = skill_cfg["effector_one_hot_idx"],
            )

        # 加载对应策略
        print(f"  加载策略...")
        policy = _load_policy(model_path, env, device)

        # ──────────────────────────────────────────────────────────────
        # 打标主循环
        # ──────────────────────────────────────────────────────────────
        # 每个 env 的状态
        env_target_idx   = list(range(min(num_envs, n_points)))   # 当前负责的 grid 点索引
        env_episode_cnt  = [0]  * num_envs                        # 已完成 episode 数
        env_hit_cnt      = [0]  * num_envs                        # 命中次数
        env_active       = [i < n_points for i in range(num_envs)] # 是否还在工作
        next_unassigned  = num_envs                                # 下一个待分配的 grid 点
        # 当前 env episode 是否发生过命中（锁存，直到该 episode 结束）
        episode_hit_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)

        completed   = 0                    # 已完成的目标点数
        total       = n_points
        t_start     = time.time()
        stable_hit_episodes = 0
        unstable_hit_episodes = 0

        # 为每个活跃 env 设置初始目标位置
        active_ids = torch.tensor(
            [i for i in range(num_envs) if i < n_points],
            device=device, dtype=torch.long
        )
        if len(active_ids) > 0:
            init_targets = torch.tensor(
                grid_points[active_ids.cpu().numpy()], device=device, dtype=torch.float32
            )
            command.set_per_env_targets(active_ids, init_targets)

        # 重置所有 env，获取初始观测
        obs = env.get_observations()

        print(f"  开始打标 ({total} 个目标点)...")
        while completed < total:
            with torch.inference_mode():
                actions = policy(obs)
            # env.step 必须在 inference_mode 块外执行，否则环境内部张量
            # 会被标记为 inference tensor，导致后续 inplace 修改报 RuntimeError
            obs, _, dones, _ = env.step(actions)

            # 命中锁存：只要本 episode 任意时刻命中过一次，就保持 True
            # 同时吸收 cumulative_hit_count 和 last_episode_had_hit，避免中途内部重采样覆盖状态
            episode_hit_latched |= (command.cumulative_hit_count > 0)
            episode_hit_latched |= command.last_episode_had_hit

            # 检查本步骤哪些 env 完成了一个 episode
            done_ids = torch.where(dones)[0].cpu().tolist()
            # True 表示由非 timeout 终止（例如摔倒）
            terminated_flags = env.unwrapped.termination_manager.terminated

            for env_id in done_ids:
                if not env_active[env_id]:
                    continue

                # 读取本 episode 是否命中（在 _resample_command 中清零前已记录）
                had_hit = bool(episode_hit_latched[env_id].item())
                fell = bool(terminated_flags[env_id].item())

                if had_hit and fell:
                    unstable_hit_episodes += 1
                if had_hit and (not fell):
                    stable_hit_episodes += 1

                if args_cli.count_fall_after_hit_as_success:
                    episode_success = had_hit
                else:
                    # 严格逻辑：仅命中且未摔倒才计入成功
                    episode_success = had_hit and (not fell)

                if episode_success:
                    env_hit_cnt[env_id] += 1
                env_episode_cnt[env_id] += 1
                # 新 episode 开始前清空锁存状态
                episode_hit_latched[env_id] = False

                if env_episode_cnt[env_id] < n_episodes:
                    # 还需继续测该目标点：_per_env_target_override 保持不变，
                    # 下次 episode reset 时目标自动设置到同一位置，无需额外操作。
                    pass
                else:
                    # 当前目标点已完成 n_episodes 次测试
                    target_idx = env_target_idx[env_id]
                    accuracy   = env_hit_cnt[env_id] / n_episodes
                    all_accuracies[target_idx, skill_idx] = accuracy
                    completed += 1

                    if completed % 500 == 0 or completed == total:
                        elapsed = time.time() - t_start
                        speed   = completed / elapsed
                        eta     = (total - completed) / speed if speed > 0 else 0
                        print(f"  进度: {completed}/{total} 点 | "
                              f"耗时 {elapsed:.0f}s | ETA {eta:.0f}s")

                    # 分配下一个目标点
                    if next_unassigned < n_points:
                        new_idx = next_unassigned
                        next_unassigned += 1
                        env_target_idx[env_id]  = new_idx
                        env_episode_cnt[env_id] = 0
                        env_hit_cnt[env_id]     = 0
                        # 设置新目标（下一个 episode reset 时生效）
                        command.set_per_env_targets(
                            torch.tensor([env_id], device=device, dtype=torch.long),
                            torch.from_numpy(grid_points[new_idx:new_idx+1]).to(device),
                        )
                    else:
                        env_active[env_id] = False

        elapsed_skill = time.time() - t_start
        mean_acc = all_accuracies[:, skill_idx].mean()
        print(f"  [{skill_name}] 完成 | 耗时 {elapsed_skill:.0f}s | 平均命中率 {mean_acc:.3f}")
        print(
            f"  [{skill_name}] 统计: 稳定命中(未摔倒)={stable_hit_episodes}, "
            f"不稳定命中(后续摔倒)={unstable_hit_episodes}"
        )

        # 保存单技能结果（支持中断后部分恢复）
        np.save(output_dir / f"{skill_name}_accuracy.npy", all_accuracies[:, skill_idx])

    # ------------------------------------------------------------------
    # 7. 生成最终标签: 各点命中率最高的技能；全为 0 则标为 stance
    # ------------------------------------------------------------------
    print("\n[打标] 生成最终标签...")
    skill_ids    = [SKILL_CONFIGS[s]["skill_id"] for s in skills_to_test]
    final_labels = np.full(n_points, STANCE_SKILL_ID, dtype=np.int32)  # 默认: stance

    for j in range(n_points):
        best_local_idx = int(np.argmax(all_accuracies[j]))
        if all_accuracies[j, best_local_idx] > 0.0:
            final_labels[j] = skill_ids[best_local_idx]

    # ------------------------------------------------------------------
    # 8. 保存所有结果
    # ------------------------------------------------------------------
    np.save(output_dir / "grid_points.npy",     grid_points)
    np.save(output_dir / "all_accuracies.npy",  all_accuracies)
    np.save(output_dir / "labels.npy",          final_labels)

    metadata = {
        "grid_spacing":   args_cli.grid_spacing,
        "n_episodes":     n_episodes,
        "num_envs":       num_envs,
        "skills_tested":  skills_to_test,
        "skill_ids":      skill_ids,
        "stance_skill_id": STANCE_SKILL_ID,
        "n_points":       n_points,
        "x_range":   [0.0,  1.2],
        "y_range":   [-0.5, 0.5],
        "z_range":   [-0.4, 0.4],
        "count_fall_after_hit_as_success": args_cli.count_fall_after_hit_as_success,
        "label_distribution": {
            str(sid): int(np.sum(final_labels == sid))
            for sid in sorted(set(skill_ids) | {STANCE_SKILL_ID})
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 9. 汇总统计
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[打标] 完成！结果保存至: {output_dir}")
    print(f"  grid_points.npy    : {n_points} 个采样点坐标")
    print(f"  all_accuracies.npy : ({n_points}, {n_skills}) 命中率矩阵")
    print(f"  labels.npy         : {n_points} 个目标点的最优技能标签")
    print(f"\n标签分布:")
    skill_name_map = {cfg["skill_id"]: name for name, cfg in SKILL_CONFIGS.items()}
    for sid, count in sorted(metadata["label_distribution"].items()):
        name = skill_name_map.get(int(sid), "unknown")
        pct  = 100.0 * count / n_points
        print(f"  [{sid}] {name:12s}: {count:5d} 点 ({pct:.1f}%)")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
