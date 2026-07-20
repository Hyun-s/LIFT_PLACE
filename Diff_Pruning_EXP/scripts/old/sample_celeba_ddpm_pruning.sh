export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

PRUNING_RATIO=$1
CUDA_DEVICE=$2

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python finetune.py \
 --config celeba.yml \
 --exp "/workspace/hyuns/src/comp/Diff-Pruning/ckpt/celeba_sample/pr${PRUNING_RATIO}" \
 --sample \
 --timesteps 100 \
 --eta 0 \
 --ni \
 --doc "sample" \
 --skip_type "uniform"  \
 --pruning_ratio "$PRUNING_RATIO" \
 --fid \
 --use_ema \
 --restore_from "/workspace/hyuns/src/comp/Diff-Pruning/ckpt/svs_celebahq_pr${PRUNING_RATIO}.pth"