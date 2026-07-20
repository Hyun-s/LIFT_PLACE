#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --qos=new_default
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH -o /workspace/hyuns/src/TinyFusion/slurm/o/%x_%j.out
#SBATCH -e /workspace/hyuns/src/TinyFusion/slurm/e/%x_%j.err



export NCCL_TIMEOUT=3600 
#/workspace/hyuns/src/TinyFusion/outputs/DiT-D7-2/D7-Learned/logit_no_mask_1/sample/outputs-DiT-D7-2-D7-Learned-logit_no_mask_1-checkpoints-0100000
export HF_HOME=/workspace/hyuns/huggingface_cache
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
# export OMP_NUM_THREADS=1    # 프로세스당 8개 스레드 사용
# export MKL_NUM_THREADS=1    # (numpy/torch에서 Intel MKL 쓰는 경우도 함께 조정)

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # count comma-separated list
  IFS=',' read -ra _g <<< "$CUDA_VISIBLE_DEVICES"; NUM_GPUS="${#_g[@]}"
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

# PRUNING_RATIO=$1
LAMBDA=$1
METHOD=$2    # "OKD" 또는 "finetune"
GROUP_SIZE=$3
ITER=$4
DEPTH=$5
# DEPTH="D7"
PATCH="2"
MODEL="DiT-"${DEPTH}"/"${PATCH}""
exp_name="DIT_IMAGENET_${METHOD}"
PTH="DiT-"${DEPTH}"-"${PATCH}""
ITER=$(printf "%07d" "$ITER")

# 포트 분리
ACCEL_PORT=$(shuf -i 4-50000 -n 1)
JUPYTER_PORT=8888
ENV="comp"
TOKEN="${ENV:-}"   # ENV 변수가 없으면 빈 문자열

# torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port $ACCEL_PORT sample_ddp.py \
#     --model ${MODEL} \
#     --ckpt       /workspace/hyuns/src/TinyFusion/outputs/DiT-D7-2/D7-Learned/patch_2-1_0109/checkpoints/${ITER}.pt \
#     --sample-dir /workspace/hyuns/src/TinyFusion/outputs/DiT-D7-2/D7-Learned/patch_2-1_0109/sample

if [ "$METHOD" == "finetune" ]; then
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port $ACCEL_PORT sample_ddp.py \
    --model ${MODEL} \
    --ckpt outputs/${PTH}/${DEPTH}-Learned/Finetuning/checkpoints/${ITER}.pt \
    --sample-dir outputs/${PTH}/${DEPTH}-Learned/Finetuning/sample --teacher

elif [["$METHOD" == "LIFT_PLACE"  ]] ; then
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port $ACCEL_PORT sample_ddp.py \
    --model ${MODEL} \
    --ckpt outputs/${PTH}/${DEPTH}-Learned/${METHOD}_${GROUP_SIZE}-${LAMBDA}/checkpoints/${ITER}.pt \
    --sample-dir outputs/${PTH}/${DEPTH}-Learned/${METHOD}_${GROUP_SIZE}-${LAMBDA}/sample

else
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port $ACCEL_PORT sample_ddp.py \
    --model ${MODEL} \
    --ckpt outputs/${PTH}/${DEPTH}-Learned/${METHOD}_${LAMBDA}/checkpoints/${ITER}.pt \
    --sample-dir outputs/${PTH}/${DEPTH}-Learned/${METHOD}_${LAMBDA}/sample
fi

