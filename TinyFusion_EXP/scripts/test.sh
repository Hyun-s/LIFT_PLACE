DEPTH="D4"
PATCH="2"
MODEL="DiT-"${DEPTH}"/"${PATCH}""

CUDA_VISIBLE_DEVICES=2 python visualize_activation.py --model ${MODEL} \
    --ckpt /workspace/hyuns/src/TinyFusion/outputs/DiT-D4-2/D4-Learned/logit_1/checkpoints/0500000.pt \
