#!/usr/bin/env bash

BASEDIR=$PWD

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1" 

CUDA_DEVICE=1
PRUNING_RATIO=$1       
LAMBDA=$2
METHOD=$3 # OKD
ITER=$4
exp_name="DIFF_CELEBA_${METHOD}"
echo "Experiment: $exp_name"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python "$BASEDIR/finetune.py" \
  --config celeba.yml \
  --exp run/finetune_final/"$exp_name"/celeba_"${PRUNING_RATIO}"_"${LAMBDA}" \
  --sample \
  --timesteps 100 \
  --eta 0 \
  --ni \
  --doc "sample" \
  --skip_type "uniform"  \
  --pruning_ratio "$PRUNING_RATIO" \
  --fid \
  --use_ema \
  --restore_from "$BASEDIR/run/finetune_final/"$exp_name"/celeba_"${PRUNING_RATIO}"_"${LAMBDA}"/logs/post_training/ckpt_"${ITER}".pth" \
cd "$BASEDIR/evals"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python prdc_score.py \
  --save-stats \
  "$BASEDIR/run/finetune_final/$exp_name/celeba_${PRUNING_RATIO}_${LAMBDA}/image_samples/images/0" \
  "$BASEDIR/run/finetune_final/$exp_name/celeba_${PRUNING_RATIO}_${LAMBDA}/prdc_stats_samples_${ITER}_cifar10.npz" \
  --device cuda:0 \
  --batch-size 256

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_gs.py \
  --sparsity "$PRUNING_RATIO" \
  --exp_name "$exp_name" \
  --real_features_fp "feats/prdc_stats_celeba.npz" \
  --fake_features_fp "$BASEDIR/run/finetune_final/$exp_name/celeba_${PRUNING_RATIO}_${LAMBDA}/prdc_stats_samples_${ITER}_cifar10.npz" \
  --nearest_k 5 \
  --update False \
  >> "$BASEDIR/run/finetune_final/$exp_name/celeba_${PRUNING_RATIO}_${LAMBDA}/eval${ITER}.txt"
