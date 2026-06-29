#!/bin/bash
#SBATCH -J mag_train                      # job name
#SBATCH -o jiahui/log/train_%j.out        # stdout
#SBATCH -e jiahui/log/train_%j.err        # stderr (Traceback 会在这里)
#SBATCH -n 1                              # 1 个任务(launcher 进程); DDP 子进程由 mp.spawn 自己起
#SBATCH -c 32                             # CPU cores (4 卡数据加载)
#SBATCH -N 1                              # 单节点
#SBATCH -p gpu-l40                        # L40 partition
#SBATCH -t 7-00:00:00                     # walltime: 7 天 (105 epoch 满训, 见下方说明)
#SBATCH --gres=gpu:L40:4                  # 4 张 L40 (和论文一致, DDP)

# ---- 环境 ----
source /cm/shared/apps/conda/23.1.0/etc/profile.d/conda.sh
conda activate /bsuscratch/jiahuizhang/envs/magician
nvidia-smi

cd /bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN

# ---- 训练 ----
# ① 首次先验证(全 8 场景, 2 epoch, 覆盖 memory replay): 去掉下行注释、注释掉满训行
# python -u jiahui/train_macarons_run.py -c train_ddp_config.json --epochs 2
# ② 正式满训 (配置里的 105 epoch):
python -u jiahui/train_macarons_run.py -c train_ddp_config.json
