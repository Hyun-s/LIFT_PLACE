




LAMBDA=$1
METHOD=$2    # "OKD" 또는 "finetune"
NSEG=$3
ITER=$4


sbatch slurmscripts/sample.sh ${LAMBDA} ${METHOD} ${NSEG} ${ITER}
sbatch slurmscripts/eval.sh ${LAMBDA} ${METHOD} ${NSEG} ${ITER}