#!/bin/bash

# Execute the Python script with the provided arguments
CUDA_VISIBLE_DEVICES=2 python prune.py \
--config "celeba.yml" \
--timesteps "100" \
--eta "0" \
--ni \
--doc "post_training" \
--skip_type "quad" \
--pruning_ratio "0.9" \
--use_ema \
--use_pretrained \
--pruner "ours" \
--save_pruned_model "run/pruned_final/celeba0.9_T=0.00.pth" \
--taylor_batch_size "64" \
--thr "0.00"