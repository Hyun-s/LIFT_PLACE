#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --qos=a6000_qos_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH -o /workspace/hyuns/src/TinyFusion/slurm/o/%x_%j.out
#SBATCH -e /workspace/hyuns/src/TinyFusion/slurm/e/%x_%j.err


export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=32    
export MKL_NUM_THREADS=32    

export TF_DISABLE_AUTOGRAPH=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eval
# PRUNING_RATIO=$1
export CUDA_VISIBLE_DEVICES=0
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


if [ "$METHOD" == "finetune" ]; then
    python evaluator.py \
    /workspace/hyuns/src/OKD_structured/evals/feats/VIRTUAL_imagenet256_labeled.npz \
    /workspace/hyuns/src/TinyFusion/outputs/${PTH}/${DEPTH}-Learned/Finetuning/sample/outputs-${PTH}-${DEPTH}-Learned-Finetuning-checkpoints-${ITER}.npz \
    >> outputs/${PTH}/${DEPTH}-Learned/Finetuning/eval_correct_${ITER}.txt

elif [["$METHOD" == "LIFT_PLACE"  ]] ; then
    python evaluator.py \
    /workspace/hyuns/src/OKD_structured/evals/feats/VIRTUAL_imagenet256_labeled.npz \
    /workspace/hyuns/src/TinyFusion/outputs/${PTH}/${DEPTH}-Learned/${METHOD}_${GROUP_SIZE}-${LAMBDA}/sample/outputs-${PTH}-${DEPTH}-Learned-${METHOD}_${GROUP_SIZE}-${LAMBDA}-checkpoints-${ITER}.npz \
    >> outputs/${PTH}/${DEPTH}-Learned/${METHOD}_${GROUP_SIZE}-${LAMBDA}/eval.txt
else
    python evaluator.py \
    /workspace/hyuns/src/OKD_structured/evals/feats/VIRTUAL_imagenet256_labeled.npz \
    /workspace/hyuns/src/TinyFusion/outputs/${PTH}/${DEPTH}-Learned/${METHOD}_${LAMBDA}/sample/outputs-${PTH}-${DEPTH}-Learned-${METHOD}_${LAMBDA}-checkpoints-${ITER}.npz \
    >> outputs/${PTH}/${DEPTH}-Learned/${METHOD}_${LAMBDA}/eval.txt
fi
