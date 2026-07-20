# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


export WANDB_MODE=offline
export HF_HOME=/workspace/hyuns/huggingface_cache
MODEL_NAME="CompVis/stable-diffusion-v1-4"
TRAIN_DATA_DIR="./data/laion_aes/preprocessed_212k" # please adjust it if needed
UNET_CONFIG_PATH="./src/unet_config"

UNET_NAME="bk_tiny" # option: ["bk_base", "bk_small", "bk_tiny"]
OUTPUT_DIR="./results/ddp_"$UNET_NAME # please adjust it if needed

BATCH_SIZE=4  # Batch size per GPU
TOTAL_BATCH_SIZE=64  # BATCH_SIZE * GRAD_ACCUMULATION = TOTAL_BATCH_SIZE
GRAD_ACCUMULATION=$((TOTAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS)))  # Dynamically calculate GRAD_ACCUMULATION
max_train_steps = max_train_steps*NUM_GPUS
StartTime=$(date +%s)
accelerate launch \
  --config_file acc_config/gpu4_config.yaml \
  --multi_gpu --num_processes ${NUM_GPUS} src/reg_kd_train_text_to_image.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $TRAIN_DATA_DIR\
  --use_ema \
  --resolution 512 --center_crop --random_flip \
  --train_batch_size $BATCH_SIZE \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate 5e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --report_to="wandb" \
  --max_train_steps=50000 \
  --seed 1234 \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --checkpointing_steps 5000 \
  --valid_steps 500 \
  --lambda_sd 1.0 --lambda_kd_output 1.0 --lambda_kd_feat 1.0 \
  --unet_config_path $UNET_CONFIG_PATH --unet_config_name $UNET_NAME \
  --output_dir $OUTPUT_DIR


EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."

