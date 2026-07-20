#!/bin/bash

# Execute the Python script with the provided arguments
python prune.py \
--config "bedroom.yml" \
--timesteps "100" \
--eta "0" \
--ni \
--doc "post_training" \
--skip_type "quad" \
--pruning_ratio "0.9" \
--use_ema \
--use_pretrained \
--pruner "ours" \
--save_pruned_model "run/pruned/bedroom_ddpm_0.9.pth" \
--taylor_batch_size "4"