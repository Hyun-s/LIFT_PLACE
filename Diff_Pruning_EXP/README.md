

<h1> [CVPR 2026] LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models </h1>
This repo based on [Diff-Pruning: Structural Pruning for Diffusion Models](https://github.com/VainF/Diff-Pruning).
For the kd loss function, refer to the functions/losses.py.

## 1. Prepare

### Requirements
```bash
pip install -r requirements.txt
```

### Download Pre-trained/Pruned celeba diffusion models for pruned initial weight.
```bash
gdown https://drive.google.com/drive/folders/1MeWya1fe2BziF96-9azk75OQ96ho-dI9?usp=sharing --folder
gdown https://drive.google.com/drive/folders/1_R0OcvwvrVwoAjUjgFjRbBeLPNU49amy?usp=drive_link --folder

mkdir -p run && cd run
gdown https://drive.google.com/drive/folders/1X9fOxTAoWgV4JTR-D-pDaVO3sxfszHdY?usp=sharing --folder
```
### prepare dataset
```bash
└── data
    └── celeba
        ├── Anno
        ├── CelebA
        ├── Eval
        └── Img
``` 

## 2. Train
### Usage
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/celeba.sh {sparsity} {lambda_kd} {method} {patch_size}
```
### Example
LIFT and PLACE sparsity 0.9 with lambda_kd 1 and group_size 16
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/celeba.sh 0.9 1 LIFT_PLACE_fitnet 16
```

## 3. Evaluation
### Requirements
```bash
gdown https://drive.google.com/drive/folders/1_JLw8ihGhQMFEI-NMVb1M0y8TpvEeJBn?usp=sharing --folder
cd evals
```


### Usage
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/celeba.sh {sparsity} {lambda_kd} {method} {ITER}
```


### Examples
Evaluate sparsity 0.3 finetuned models (100k)
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/sample_eval.sh 0.3 1 finetune 100000
```


## Cite this work
If you found this repository useful, please consider giving a star and citation:
```bibtex
@inproceedings{han2026lift,
  title={LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models},
  author={Han, Hyunsoo and Yeo, Sangyeop and Yoo, Jaejun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5564--5573},
  year={2026}
}
```
