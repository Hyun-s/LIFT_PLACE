#!/bin/bash
#SBATCH --job-name=bk_base
#SBATCH --partition=A6000 # A6000 , A100 , R4090 , R3090
#SBATCH --qos=new_default # a100_qos_4   , gpu_qos_8, a6000_qos_8
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o /workspace/hyuns/src/BK-SDM/slurm/o/%x_%j.out
#SBATCH -e /workspace/hyuns/src/BK-SDM/slurm/e/%x_%j.err

# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------


# # Optional: keep CPU libs tame inside dataloader workers
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export TORCH_NUM_THREADS=1
# export TORCH_INTEROP_THREADS=1


# Master address/port for distributed (safe dynamic port 49152–65535)
export MASTER_ADDR="${MASTER_ADDR:-$(hostname -s)}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  while :; do
    p=$(shuf -i 49152-65535 -n 1)
    (exec 3<>"/dev/tcp/127.0.0.1/$p") >/dev/null 2>&1 && { exec 3>&- 3<&-; continue; } || {
      export MASTER_PORT="$p"; break;
    }
  done
fi

# Respect Slurm's GPU allocation (don't override CUDA_VISIBLE_DEVICES)
if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NUM_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # count comma-separated list
  IFS=',' read -ra _g <<< "$CUDA_VISIBLE_DEVICES"; NUM_GPUS="${#_g[@]}"
else
  NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

if [[ -z "${BATCH_SIZE_PER_GPU:-}" ]]; then
  # nvidia-smi로 이름과 총 메모리(MiB) 조회
  mapfile -t GPU_INFO < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)

  # 기본값(미지정/기타 기종)
  BS_MIN=8

  for line in "${GPU_INFO[@]}"; do
    gpu_name="${line%%,*}"; mem_mb="${line##*, }"
    echo $gpu_name
    mem_gb=$(( mem_mb / 1024 ))

    # 기종/용량 매칭 (필요시 숫자 조정)
    if [[ "$gpu_name" == *A100* && $mem_gb -ge 75 ]]; then
      bs=32        # A100 80GB
    elif [[ "$gpu_name" == *A6000* ]]; then
      bs=32        # RTX A6000 48GB
    elif [[ "$gpu_name" == *4090* ]]; then
      bs=8         # RTX 4090 24GB
    elif [[ "$gpu_name" == *3090* ]]; then
      bs=8         # RTX 3090 24GB
    else
      # 메모리 기준 보수적 추정 (원하면 조정)
      if   (( mem_gb >= 70 )); then bs=16
      elif (( mem_gb >= 40 )); then bs=12
      else                         bs=8
      fi
    fi

    # 혼합기종이면 가장 작은 값으로
    # (( bs < BS_MIN )) && BS_MIN=$bs
  done

  BATCH_SIZE_PER_GPU=$bs
fi

# grad_accum = ceil(TOTAL / (per-GPU * NGPU))
# denom=$(( BATCH_SIZE_PER_GPU * NUM_GPUS ))
# GRAD_ACCUMULATION=$(( (TOTAL_BATCH_SIZE + denom - 1) / denom ))
# (( GRAD_ACCUMULATION < 1 )) && GRAD_ACCUMULATION=1

# echo "[AUTO-BATCH] per-GPU BATCH_SIZE=${BATCH_SIZE_PER_GPU}, NUM_GPUS=${NUM_GPUS}, GRAD_ACCUMULATION=${GRAD_ACCUMULATION} (≈TOTAL=$((BATCH_SIZE_PER_GPU*NUM_GPUS*GRAD_ACCUMULATION)))"
# ================================================================

# 이후 accelerate 인자에 사용
BATCH_SIZE="$BATCH_SIZE_PER_GPU"


# =========================== Safety & Args ===========================
set -Eeuo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <UNET_NAME> <LOSS_TYPE> <PATCH>"
  echo "  UNET_NAME ∈ [bk_base,bk_small,bk_tiny,C_base,C_small,C_tiny,C_micro,C_extreme]"
  exit 2
fi

UNET_NAME="$1"    # ["bk_base","bk_small","bk_tiny","C_base","C_small","C_tiny","C_micro","C_extreme"]
LOSS_TYPE="$2"
PATCH="$3"

EXTRA_ARGS=()
if [[ "$UNET_NAME" == *C* ]]; then
  EXTRA_ARGS+=(--channel_mapping)
fi

if [[ "$UNET_NAME" == *bk* ]]; then
  EXTRA_ARGS+=(--use_copy_weight_from_teacher)
fi

ACC_ARGS=( --config_file "acc_config/gpu${NUM_GPUS}_config_DS.yaml" )
if [ "$NUM_GPUS" -gt 1 ]; then
  ACC_ARGS+=( --multi_gpu --num_processes "$NUM_GPUS" )
fi

# =========================== Environment ============================

export WANDB_MODE=offline
export HF_HOME=/workspace/hyuns/huggingface_cache
# MODEL_NAME="CompVis/stable-diffusion-v1-4"

MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
# TRAIN_DATA_DIR="./data/laion_aes/preprocessed_212k"   # adjust if needed
TRAIN_DATA_DIR="/data/hyuns/preprocessed_212k"
UNET_CONFIG_PATH="./src/unet_config_v2-base"
OUTPUT_DIR="./results_v2/${UNET_NAME}_${LOSS_TYPE}_${PATCH}"


# Batch/accumulation
# BATCH_SIZE=16                        # per-GPU
BATCH_SIZE="$BATCH_SIZE_PER_GPU"
TOTAL_BATCH_SIZE=128                 # BATCH_SIZE * GRAD_ACCUMULATION * NUM_GPUS
# ceil division to keep >=1 even if not divisible exactly
denom=$(( BATCH_SIZE * NUM_GPUS ))
GRAD_ACCUMULATION=$(( (TOTAL_BATCH_SIZE) / denom ))


echo "===== Launch Config ====="
echo "UNET_NAME=$UNET_NAME  LOSS_TYPE=$LOSS_TYPE  PATCH=$PATCH"
echo "NUM_GPUS=$NUM_GPUS  BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUMULATION=$GRAD_ACCUMULATION"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo "========================="


# =========================== Train ================================
StartTime=$(date +%s)

accelerate launch \
  "${ACC_ARGS[@]}"\
  src/kd_t2i.py \
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --train_data_dir "$TRAIN_DATA_DIR" \
  --use_ema \
  --resolution 512 --center_crop --random_flip \
  --train_batch_size "$BATCH_SIZE" \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate 5e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --report_to="wandb" \
  --max_train_steps=50000 \
  --seed 1234 \
  --gradient_accumulation_steps "$GRAD_ACCUMULATION" \
  --checkpointing_steps 5000 \
  --valid_steps 10000 \
  --lambda_sd 1.0 --lambda_kd_output 1.0 --lambda_kd_feat 1.0 \
  --unet_config_path "$UNET_CONFIG_PATH" --unet_config_name "$UNET_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --loss_type "$LOSS_TYPE" --patch_size "$PATCH" \
  "${EXTRA_ARGS[@]}"

EndTime=$(date +%s)
echo "** KD training takes $((EndTime - StartTime)) seconds."



