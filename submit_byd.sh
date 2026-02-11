#!/bin/bash
#BSUB -m "hostm-10"
#BSUB -J g1_fight_train
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 72:00
#BSUB -o /home/rashare/yangkun/whole_body_tracking/logs/job_%J.out
#BSUB -e /home/rashare/yangkun/whole_body_tracking/logs/job_%J.err

# --- 1. 环境清理与路径设置 ---
module purge
# 加载基础 CUDA 环境，Isaac Sim 需要它
module load cuda/12.1 2>/dev/null

# 显式初始化 Conda 并激活你针对 BeyondMimic 的环境
source /home/rashare/miniconda3/etc/profile.d/conda.sh
conda activate beyondmimic_yk

# --- 2. 关键环境变量设置 ---
# 设置 WandB 归属地与项目
export WANDB_ENTITY="HCL_Humanoid"
export WANDB_PROJECT="Beyondmimic"
# 使用你刚才生成的最新 API Key
export WANDB_API_KEY=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
export WANDB_MODE=online

# 修复某些系统下的线程冲突问题
export MKL_THREADING_LAYER=GNU

# --- 3. 定位到项目根目录 ---
cd /home/rashare/yangkun/whole_body_tracking || exit

# --- 4. 关键修复：手动添加 IsaacLab 路径 ---
# 这一步确保 Python 能找到 'isaaclab' 模块
export PYTHONPATH=$PYTHONPATH:/home/rashare/yangkun/IsaacLab/source/isaaclab
# 同时也把当前项目路径加进去
export PYTHONPATH=$PYTHONPATH:/home/rashare/yangkun/whole_body_tracking/source/whole_body_tracking

# --- 5. 启动正式训练 ---
# 注意：IsaacLab 必须通过项目外部的 isaaclab.sh 脚本启动，以确保内部的 Python 路径和仿真引擎正确挂载
/home/rashare/yangkun/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
   --task Tracking-Flat-G1-v0 \
   --registry_name 2909438315-shenzhen-technology-university-business-school-org/wandb-registry-Motions/fight1_subject2:v0 \
   --headless \
   --logger wandb \
   --num_envs 4096 \
   --max_iterations 50000 \
   --run_name g1_fight_official_v1