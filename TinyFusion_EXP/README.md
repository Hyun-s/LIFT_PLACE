

<h1> [CVPR 2026] LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models </h1>
This repository is based on TinyFusion: Diffusion Transformers Learned Shallow.

## 1. Prepare

### Requirements
Install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

### Prepare Experiment Settings.
The experimental setup follows the BK-SDM baseline.
TinyFusion [https://github.com/VainF/TinyFusion] 의 preparation을 참조해서 준비할 수 있음.


## 2. Train with slurm

### Usage
Run the training script using sbatch with your desired UNet architecture, loss type, and group size.
```bash
sbatch scripts/train_d14.sh <KD-LAMBDA> <LOSS_TYPE> <GROUP_SIZE>
```

### Example
To train LIFT and PLACE on BK-Tiny with a group_size of 16:
```bash
sbatch scripts/train_d14.sh 1 LIFT_PLACE 16
```

## 3. Evaluation
### Usage
Models are saved with an ID formatted as ${UNET_NAME}_${LOSS_TYPE}_${GROUP_SIZE} (or PATCH). Use this MODEL_ID and the target iteration step to run the evaluation.
```bash
sbatch scripts/sample.sbatch <KD-LAMBDA> <LOSS_TYPE> <GROUP_SIZE> <ITER> <DEPTH>
sbatch scripts/eval.sbatch <KD-LAMBDA> <LOSS_TYPE> <GROUP_SIZE> <ITER> <DEPTH>
```


### Examples
```bash
sbatch scripts/sample.sbatch 1 LIFT_PLACE 16 500000 D14
sbatch scripts/eval.sbatch 1 LIFT_PLACE 16 500000 D14
```


## Cite this work
If you found this repository useful, please consider giving a star and citation:
```bash
@inproceedings{han2026lift,
  title={LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models},
  author={Han, Hyunsoo and Yeo, Sangyeop and Yoo, Jaejun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5564--5573},
  year={2026}
}
```
