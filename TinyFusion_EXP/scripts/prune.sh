#!/bin/bash
#SBATCH --job-name=prune_dit
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o /workspace/hyuns/src/TinyFusion/slurm/o/%x_%j.out
#SBATCH -e /workspace/hyuns/src/TinyFusion/slurm/e/%x_%j.err
export OMP_NUM_THREADS=16    # 프로세스당 8개 스레드 사용
export MKL_NUM_THREADS=16   # (numpy/torch에서 Intel MKL 쓰는 경우도 함께 조정)

export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export HF_HOME=/workspace/hyuns/huggingface_cache

# PRUNING_RATIO=$1
LAMBDA=$1
METHOD=$2    # "OKD" 또는 "finetune"
exp_name="DIFF_CHURCH_${METHOD}"

# 포트 분리
ACCEL_PORT=$(shuf -i 4-50000 -n 1)
JUPYTER_PORT=8888
ENV="comp"
TOKEN="${ENV:-}"   # ENV 변수가 없으면 빈 문자열

echo "→ accelerate port: ${ACCEL_PORT}"
echo "→ jupyter lab port: ${JUPYTER_PORT}"


# torchrun --nnodes=1 --nproc_per_node=1 prune_by_learning.py \
#   --model DiT-XL-1-7 \
#   --load-weight pretrained/DiT-XL-2-256x256.pt \
#   --data-path data/imagenet_encoded \
#   --epochs 1 \
#   --global-batch-size 128 \
#   --delta-w \
#   --lora \
#   --save-model outputs/pruned/DiT-D14-Learned.pt


# torchrun --nnodes=1 --nproc_per_node=4 prune_by_learning.py \
#   --model DiT-XL-1-2 \
#   --load-weight pretrained/DiT-XL-2-256x256.pt \
#   --data-path data/imagenet_encoded \
#   --epochs 1 \
#   --global-batch-size 128 \
#   --delta-w \
#   --lora \
#   --save-model outputs/pruned/DiT-D14-Learned.pt

# torchrun --nnodes=1 --nproc_per_node=8 prune_by_learning.py \
#   --model DiT-D14-1-2 \
#   --load-weight pretrained/TinyDiT-D14-MaskedKD-500K.pt \
#   --data-path /data/hyuns/imagenet_encoded/imagenet_encoded \
#   --epochs 1 \
#   --global-batch-size 128 \
#   --delta-w \
#   --lora \
#   --save-model outputs/pruned/DiT-D7-Learned_from_D14.pt  



# torchrun --nnodes=1 --nproc_per_node=4 prune_by_learning.py \
# --model DiT-D7-1-2 \
# --load-weight /workspace/hyuns/src/TinyFusion/outputs/DiT-D7-2/D7-Learned/logit_1/checkpoints/0500000.pt \
# --data-path /data/hyuns/imagenet_encoded/imagenet_encoded \
# --epochs 1 \
# --global-batch-size 128 \
# --delta-w \
# --lora \
# --save-model outputs/pruned/DiT-D4-Learned_from_D7.pt  
torchrun --nnodes=1 --nproc_per_node=4 --master_port $ACCEL_PORT prune_by_learning.py \
--model DiT-XL-1-7 \
--load-weight /workspace/hyuns/src/TinyFusion/pretrained/DiT-XL-2-256x256.pt \
--data-path /data/hyuns/imagenet_encoded/imagenet_encoded \
--epochs 1 \
--global-batch-size 128 \
--delta-w \
--lora \
--save-model outputs/pruned/DiT-D4-Learned_from_XL.pt  


# torchrun --nproc_per_node=2 prune_by_channel.py --pruning_ratio 0.75 --head_pruning_ratio 0.75