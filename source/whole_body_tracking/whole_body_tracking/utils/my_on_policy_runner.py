import os
import importlib
import warnings
import statistics
import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    _KNOWN_CLASS_REGISTRY = {
        "PPO": PPO,
        "ActorCritic": ActorCritic,
        "ActorCriticRecurrent": ActorCriticRecurrent,
    }

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            try:
                export_motion_policy_as_onnx(
                    self.env.unwrapped,
                    self.alg.policy,
                    normalizer=getattr(self, "obs_normalizer", None),
                    path=policy_path,
                    filename=filename,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
            except Exception as exc:
                # ONNX export should not break long-running training.
                print(f"[WARN] ONNX export skipped: {exc}")

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    def _build_train_value_diagnostics(self, locs: dict) -> dict[str, float]:
        """构建 iteration 级别的奖励/价值诊断指标。

        指标说明：
        - mean_episode_reward:
          来自 rewbuffer（最近完成的 episode 总奖励均值），是“真实整回合累计奖励”。
        - mean_episode_length_steps:
          来自 lenbuffer（最近完成的 episode 步数均值）。
        - mean_episode_length_s:
          mean_episode_length_steps 乘以 env.step_dt，单位秒。
        - rollout_return_mean:
          来自 PPO rollout buffer 的 returns 均值（GAE/bootstrap 之后的学习目标），
          这是 critic 要拟合的目标，不等同于完整 episode 总奖励。
        - value_prediction_mean:
          来自 PPO rollout buffer 的 values 均值（critic 当前预测）。
        - return_value_gap:
          = rollout_return_mean - value_prediction_mean，
          用来观察 critic 是低估（>0）还是高估（<0）当前数据分布下的回报目标。
        """
        diagnostics: dict[str, float] = {}

        # rewbuffer: 仅统计“本 iteration 内结束的 episode”的总奖励。
        # 因此它更接近任务表现（例如是否长期为负），但受 episode 终止频率影响。
        rewbuffer = locs.get("rewbuffer", None)
        if rewbuffer is not None and len(rewbuffer) > 0:
            diagnostics["mean_episode_reward"] = float(statistics.mean(rewbuffer))

        # lenbuffer: 仅统计“本 iteration 内结束的 episode”的步数。
        # 用于观察平均存活时长（falling out 是否减少）。
        lenbuffer = locs.get("lenbuffer", None)
        if lenbuffer is not None and len(lenbuffer) > 0:
            mean_len = float(statistics.mean(lenbuffer))
            diagnostics["mean_episode_length_steps"] = mean_len
            # Try wrapper, then unwrapped env, then cfg fallback.
            step_dt = getattr(self.env, "step_dt", None)
            if step_dt is None and hasattr(self.env, "unwrapped"):
                step_dt = getattr(self.env.unwrapped, "step_dt", None)
            if step_dt is None and hasattr(self.env, "cfg"):
                cfg = getattr(self.env, "cfg", None)
                if cfg is not None and hasattr(cfg, "sim") and hasattr(cfg, "decimation"):
                    step_dt = float(cfg.sim.dt) * float(cfg.decimation)
            if step_dt is not None:
                diagnostics["mean_episode_length_s"] = mean_len * float(step_dt)

        # storage.returns / storage.values: 来自当前 iteration rollout 的逐时间步张量，
        # 其均值更接近“优化过程的学习信号”，而不是整回合真实累计值。
        storage = getattr(self.alg, "storage", None)
        if storage is not None and hasattr(storage, "returns") and hasattr(storage, "values"):
            return_mean = float(storage.returns.mean().item())
            value_mean = float(storage.values.mean().item())
            diagnostics["rollout_return_mean"] = return_mean
            diagnostics["value_prediction_mean"] = value_mean
            diagnostics["return_value_gap"] = return_mean - value_mean

        return diagnostics

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """默认日志 + 额外训练诊断。

        额外诊断会同时写入 wandb/tensorboard 的 `Metrics/train/*`，
        并被注入主日志块（和 Episode_Reward/Metrics 同区域输出），
        便于直接对比“真实奖励表现”与“critic 拟合状态”。
        """
        diagnostics = self._build_train_value_diagnostics(locs)
        if len(diagnostics) > 0 and ("ep_infos" in locs) and len(locs["ep_infos"]) > 0:
            # 将新增指标注入到 ep_infos，复用上游日志排序与打印逻辑。
            # 这样不会在主日志块后额外追加“第二段打印”。
            for key, value in diagnostics.items():
                locs["ep_infos"][0][f"Metrics/train/{key}"] = torch.tensor([value], device=self.device)

        super().log(locs, width=width, pad=pad)

        if self.disable_logs:
            return

        if len(diagnostics) == 0:
            return

        # 持久化到 logger 后端（wandb/tensorboard）。
        for key, value in diagnostics.items():
            self.writer.add_scalar(f"Metrics/train/{key}", value, locs["it"])

    @staticmethod
    def _resolve_class(class_name: str):
        """Resolve class by short-name registry or full dotted path."""
        if class_name in MotionOnPolicyRunner._KNOWN_CLASS_REGISTRY:
            return MotionOnPolicyRunner._KNOWN_CLASS_REGISTRY[class_name]
        if "." in class_name:
            module_name, cls_name = class_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, cls_name)

        # Fallback: try canonical rsl_rl namespaces by short name.
        for module_name in ("rsl_rl.algorithms", "rsl_rl.modules"):
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                return getattr(module, class_name)

        known = sorted(MotionOnPolicyRunner._KNOWN_CLASS_REGISTRY.keys())
        raise NameError(
            f"无法解析 class_name='{class_name}'。"
            f"请使用完整路径或已注册短名。已注册: {known}"
        )

    def _construct_algorithm(self, obs):
        """Construct PPO algorithm with support for custom dotted-path policy classes."""
        # Keep the same behavior as upstream runner.
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        policy_cfg = dict(self.policy_cfg)
        alg_cfg = dict(self.alg_cfg)

        actor_class_name = policy_cfg.pop("class_name")
        alg_class_name = alg_cfg.pop("class_name")
        actor_critic_class = self._resolve_class(actor_class_name)
        alg_class = self._resolve_class(alg_class_name)

        actor_critic = actor_critic_class(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(
            self.device
        )
        alg = alg_class(actor_critic, device=self.device, **alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        # Re-resolve observation groups before parent constructor, same as upstream behavior.
        obs = env.get_observations()
        default_sets = ["policy", "critic"]
        alg_cfg = train_cfg.get("algorithm", {})
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        obs_groups_cfg = train_cfg.get("obs_groups")
        if obs_groups_cfg is None:
            obs_groups_cfg = {"policy": ["policy"], "critic": ["critic"]}
        train_cfg["obs_groups"] = resolve_obs_groups(obs, obs_groups_cfg, default_sets)

        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
