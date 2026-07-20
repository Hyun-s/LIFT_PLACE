#!/bin/bash
#SBATCH --job-name=t2i-reg-kd-sample
#SBATCH --partition=R3090
#SBATCH --qos=new_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -o /workspace/hyuns/src/BK-SDM/slurm/o/%x_%j.out
#SBATCH -e /workspace/hyuns/src/BK-SDM/slurm/e/%x_%j.err

export HF_HOME=/workspace/hyuns/huggingface_cache
MODEL_ID="${1:-}"
ITER="${2:-}"

if [[ -z "$MODEL_ID" ]]; then
  echo "Usage: $0 <MODEL_ID> [ITER|None]" >&2
  exit 1
fi

if [[ $MODEL_ID == nota-ai* \
      || $MODEL_ID == stabilityai* \
      || $MODEL_ID == CompVis* ]]; then
  BASE="${MODEL_ID}"
else
  BASE="./results_v2/${MODEL_ID}"
fi
echo "$BASE"
# ITER가 비었거나 "None"이면 기본 저장 경로 사용
if [[ -z "$ITER" || "$ITER" == "None" ]]; then
  python3 src/generate.py --model_id "${BASE}" --save_dir "${BASE}"
  bash scripts/eval_scores_v2.sh ${MODEL_ID} #>> ./results/${MODEL_ID}/evals.txt
elif [[ $MODEL_ID == nota-ai* \
      || $MODEL_ID == stabilityai* \
      || $MODEL_ID == CompVis* ]]; then
    python3 src/generate.py --model_id "${BASE}" --save_dir "./results_v2/${BASE}"
else
  ITER="checkpoint-${ITER}"
  python3 src/generate.py --model_id "${BASE}" --save_dir "${BASE}/${ITER}" --unet_path "${BASE}/${ITER}"
  bash scripts/eval_scores_v2.sh ${MODEL_ID}/${ITER} #>> ./results/${MODEL_ID}/evals.txt
fi

# bash scripts/eval_scores.sh ${MODEL_ID} #>> ./results/${MODEL_ID}/evals.txt