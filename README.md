# [CVPR 2026] LIFT and PLACE: A Simple, Stable, and Effective Knowledge Distillation Framework for Lightweight Diffusion Models

### [Arxiv](https://arxiv.org/pdf/2605.19729) / [Project Page](https://hyun-s.github.io/LIFT_PLACE_site/)

Authors: Hyunsoo Han, Sangyeop Yeo, [Jaejun Yoo](https://scholar.google.co.kr/citations?hl=en&user=7NBlQw4AAAAJ)

---
![imgs](assets/teaser.SVG)

---
## Overview
A larger teacher is not always a better teacher for lightweight diffusion models. As the capacity gap between the teacher and student increases, conventional knowledge distillation becomes harder to optimize, less stable, and less effective.

We propose LIFT and PLACE, a simple coarse-to-fine knowledge distillation framework for lightweight diffusion models.

- **LIFT (LInear FiTting-based Distillation)** decomposes the distillation error into Coarse-Easy and Fine-Hard components. It first aligns low-order statistical differences and then gradually shifts toward fine-grained refinement.
- **PLACE (Piecewise Local Adaptive Coefficient Estimation)** addresses spatially non-uniform distillation errors by grouping output elements according to their difficulty and providing locally adaptive guidance.

Our framework introduces no additional parameters or inference overhead and can be applied to a wide range of diffusion models.

## Codes for Diff-Pruning, BK-SDM and TinyFusion
Please refer following codes for each model.
- DDPM: [Diff-Pruning](Diff_Pruning_EXP/)
- Stable Diffusion: [BK-SDM](BK_SDM_EXP/)
- DiT: [TinyFusion](TinyFusion_EXP/)


## Acknowledgements
We sincerely thank the authors of Diff-Pruning, BK-SDM, and TinyFusion for open-sourcing their excellent codebases and providing strong baselines for generative model compression.

Their contributions provided a solid foundation for developing and evaluating this work across DDPM, Stable Diffusion, and Diffusion Transformer architectures.

**Diff-Pruning**
```bibtex
@inproceedings{fang2023structural,
  title={Structural pruning for diffusion models},
  author={Gongfan Fang and Xinyin Ma and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
```




**BK-SDM**
```bibtex
@inproceedings{kim2024bk,
  title={Bk-sdm: A lightweight, fast, and cheap version of stable diffusion},
  author={Kim, Bo-Kyeong and Song, Hyoung-Kyu and Castells, Thibault and Choi, Shinkook},
  booktitle={European Conference on Computer Vision},
  pages={381--399},
  year={2024},
  organization={Springer}}
```


**TinyFusion**
```bibtex
@inproceedings{fang2025tinyfusion,
  title={Tinyfusion: Diffusion transformers learned shallow},
  author={Fang, Gongfan and Li, Kunjun and Ma, Xinyin and Wang, Xinchao},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18144--18154},
  year={2025}}
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
