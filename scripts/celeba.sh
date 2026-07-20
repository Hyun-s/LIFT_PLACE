#!/bin/bash
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1

PRUNING_RATIO=$1
LAMBDA=$2
METHOD=$3 
PATCH=${4:-0}
exp_name="DIFF_CELEBA_${METHOD}"

# 포트 분리
ACCEL_PORT=$(shuf -i 4-50000 -n 1)
JUPYTER_PORT=8888
ENV="comp"
TOKEN="${ENV:-}"   # ENV 변수가 없으면 빈 문자열

# Not KD
if [[ "$METHOD" == "finetune" ]]; then
    accelerate launch \
        --main_process_port ${ACCEL_PORT} finetune.py \
        --config celeba.yml \
        --timesteps 100 \
        --eta 0 \
        --ni \
        --exp run/finetune_final/"${exp_name}"/celeba_"${PRUNING_RATIO}"_"${LAMBDA}" \
        --doc post_training \
        --skip_type uniform \
        --pruning_ratio "${PRUNING_RATIO}" \
        --patch "${PATCH}" \
        --use_ema \
        --load_pruned_model "run/pruned_final/celeba${PRUNING_RATIO}_T=0.00.pth" --loss_type "${METHOD}" 
# KD
else
    accelerate launch \
        --main_process_port ${ACCEL_PORT} finetune.py \
        --config celeba.yml \
        --timesteps 100 \
        --eta 0 \
        --ni \
        --exp run/finetune_final/"${exp_name}"/celeba_"${PRUNING_RATIO}"_"${LAMBDA}" \
        --doc post_training \
        --skip_type uniform \
        --pruning_ratio "${PRUNING_RATIO}" \
        --patch "${PATCH}" \
        --use_ema \
        --kd \
        --lamb "${LAMBDA}" \
        --loss_type "${METHOD}" \
        --load_pruned_model "run/pruned_final/celeba${PRUNING_RATIO}_T=0.00.pth" 
fi
