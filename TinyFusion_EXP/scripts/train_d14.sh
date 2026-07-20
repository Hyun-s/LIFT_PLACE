#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --qos=a6000_qos_8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH -o /workspace/hyuns/src/TinyFusion/slurm/o/%x_%j.out
#SBATCH -e /workspace/hyuns/src/TinyFusion/slurm/e/%x_%j.err
####################
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL #너무 많으면 init, graph, net만
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SHM_DISABLE=1 #공유메모리 이슈 방지
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET
####################



export OMP_NUM_THREADS=8    # 프로세스당 8개 스레드 사용
export MKL_NUM_THREADS=8    # (numpy/torch에서 Intel MKL 쓰는 경우도 함께 조정)
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8


export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1

export HF_HOME=/workspace/hyuns/huggingface_cache
export WANDB_MODE=offline
# PRUNING_RATIO=$1
LAMBDA=$1
METHOD=$2    # "OKD" 또는 "finetune"
GROUP_SIZE=$3
DEPTH="D14"
PATCH="2"
MODEL="DiT-"${DEPTH}"/"${PATCH}""
exp_name="DIT_IMAGENET_${METHOD}"


if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # count comma-separated list
  IFS=',' read -ra _g <<< "$CUDA_VISIBLE_DEVICES"; NUM_GPUS="${#_g[@]}"
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
fi


# data_path="data/imagenet_encoded"
echo "Using data path: $data_path"


# 포트 분리
ACCEL_PORT=$(shuf -i 49152-50000 -n 1)
JUPYTER_PORT=8888
ENV="comp"
TOKEN="${ENV:-}"   # ENV 변수가 없으면 빈 문자열


if [ "$METHOD" == "finetune" ]; then
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port $ACCEL_PORT train.py --model ${MODEL} \
    --load-weight outputs/pruned/DiT-${DEPTH}-Learned.pt \
    --data-path ${data_path} --epochs 100 \
    --prefix ${DEPTH}-Learned/Finetuning



elif [[ "$METHOD" == "LIFT_PLACE" \
  ]]; then
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port $ACCEL_PORT train_masked_kd.py --model ${MODEL} \
    --load-weight outputs/pruned/DiT-${DEPTH}-Learned.pt \
    --data-path ${data_path} --epochs 100 \
    --prefix ${DEPTH}-Learned/${METHOD}_${GROUP_SIZE}-${LAMBDA} \
    --lamb ${LAMBDA} --loss_type ${METHOD} --nseg ${GROUP_SIZE} \
    --teacher DiT-XL/2 \
    --load-teacher pretrained/DiT-XL-2-256x256.pt

else
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port $ACCEL_PORT train_masked_kd.py --model ${MODEL} \
    --load-weight outputs/pruned/DiT-${DEPTH}-Learned.pt \
    --data-path ${data_path} --epochs 100 \
    --prefix ${DEPTH}-Learned/${METHOD}_${LAMBDA} \
    --lamb ${LAMBDA} --loss_type ${METHOD} \
    --teacher DiT-XL/2 \
    --load-teacher pretrained/DiT-XL-2-256x256.pt
fi


