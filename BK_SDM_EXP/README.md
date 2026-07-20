

<h1> [CVPR 2026] LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models </h1>
This repository is based on BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion.

## 1. Prepare

### Requirements
Install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

### Prepare Experiment Settings.
The experimental setup follows the BK-SDM baseline.

You can download the training data using the following command:
```bash
bash scripts/get_laion_data.sh preprocessed_212k
``` 

You can download the evaluation data using the following command:
```bash
bash scripts/get_mscoco_files.sh

# ./data/mscoco_val2014_30k/metadata.csv: 30K prompts from the MS-COCO validation set (used in '(2)')  
# ./data/mscoco_val2014_41k_full/real_im256.npz: FID statistics of 41K real images (used in '(3)')
``` 


## 2. Train with slurm

### Usage
Run the training script using sbatch with your desired UNet architecture, loss type, and group size.
```bash
sbatch scripts/train_main_sd2.sh <UNET_NAME> <LOSS_TYPE> <GROUP>
```

### Example
To train LIFT and PLACE on BK-Tiny with a group_size of 16:
```bash
sbatch scripts/train_main_sd2.sh bk_tiny LIFT_PLACE 16
```

## 3. Evaluation
### Usage
Models are saved with an ID formatted as ${UNET_NAME}_${LOSS_TYPE}_${GROUP} (or PATCH). Use this MODEL_ID and the target iteration step to run the evaluation.
```bash
sbatch sample_v2.sbatch <MODEL_ID> <ITER>
```


### Examples
```bash
sbatch sample_v2.sbatch bk_tiny_LIFT_PLACE_16 50000
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
