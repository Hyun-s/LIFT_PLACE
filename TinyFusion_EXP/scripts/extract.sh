#!/bin/bash
#SBATCH --job-name=prepare_in
#SBATCH --partition=R3090
#SBATCH --qos=gpu_qos_1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH -o /workspace/hyuns/src/TinyFusion/slurm/o/%x_%j.out
#SBATCH -e /workspace/hyuns/src/TinyFusion/slurm/e/%x_%j.err
export OMP_NUM_THREADS=32    # 프로세스당 8개 스레드 사용
export MKL_NUM_THREADS=32    # (numpy/torch에서 Intel MKL 쓰는 경우도 함께 조정)

export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1


torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-XL/2 --data-path /nas1/1000_Members/hyuns/imagenet/imagenet/train --features-path /data/hyuns/datasets/imagenet_encoded