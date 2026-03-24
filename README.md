# MSAN_Integ

This work is published in *"MSAN: Multiscale Self-Attention Network for Pansharpening"*, *Pattern Recognition*, 2025.



## Directory

- `msan/model.py`: MSAN main model (interface: `spectral_num`, `spatial_num`)
- `msan/losses.py`: training loss (`L1 + MSE`, consistent with the original core logic)
- `msan/msan_swin.py`: Swin-related structures
- `msan/msan_models_others.py`: attention and auxiliary modules
- `dataset_npz.py`: standalone `.npz` dataset loader
- `train.py`: training entry
- `test.py`: testing entry

## Data format

Each sample is one `.npz` file with the following keys:

- `lms`: `[C, H, W]`
- `pan`: `[1, H, W]`
- `gt`: `[C, H, W]`

Example:

```python
import numpy as np
np.savez("sample_00001.npz", lms=lms, pan=pan, gt=gt)
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --train_dir /path/to/train_npz --epochs 50 --batch_size 8 --spectral_num 4 --spatial_num 1
```

If needed for memory or numerical stability, disable AMP:

```bash
python train.py --train_dir /path/to/train_npz --spectral_num 4 --no_amp
```

## Testing

```bash
python test.py --test_dir /path/to/test_npz --ckpt checkpoints/msan_epoch_050.pth --spectral_num 4 --spatial_num 1
```

Predictions are saved to `predictions/*.npy` by default.

## Citation

If you use MSAN in your work, please cite:

```bibtex
@article{luMSANMultiscaleSelfattention2025,
  title = {{{MSAN}}: {{Multiscale}} Self-Attention Network for Pansharpening},
  shorttitle = {{{MSAN}}},
  author = {Lu, Hangyuan and Yang, Yong and Huang, Shuying and Liu, Rixian and Guo, Huimin},
  date = {2025-06},
  journaltitle = {Pattern Recognition},
  shortjournal = {Pattern Recognition},
  volume = {162},
  pages = {111441},
  issn = {00313203},
  doi = {10.1016/j.patcog.2025.111441},
  url = {https://linkinghub.elsevier.com/retrieve/pii/S0031320325001013},
  urldate = {2025-02-26},
  abstract = {Effective extraction of spectral–spatial features from multispectral (MS) and panchromatic (PAN) images is critical for high-quality pansharpening. However, existing deep learning methods often overlook local misalignment and struggle to integrate local and long-range features effectively, resulting in spectral and spatial distortions. To address these challenges, this paper proposes a refined detail injection model that adaptively learns injection coefficients using long-range features. Building upon this model, a multiscale self-attention network (MSAN) is proposed, consisting of a feature extraction branch and a self-attention mechanism branch. In the former branch, a two-stage multiscale convolution network is designed to fully extract detail features with multiple receptive fields. In the latter branch, a streamlined Swin Transformer (SST) is proposed to efficiently generate multiscale self-attention maps by learning the correlation between local and long-range features. To better preserve spectral–spatial information, a revised Swin Transformer block is proposed by incorporating spectral and spatial attention within the block. The obtained self-attention maps from SST serve as the injection coefficients to refine the extracted details, which are then injected into the upsampled MS image to produce the final fused image. Experimental validation demonstrates the superiority of MSAN over traditional and state-of-the-art methods, with competitive efficiency. The code of this work will be released on GitHub once the paper is accepted.},
  langid = {english},
  keywords = {Multiscale, Pansharpening, Self-attention, Swin Transformer}
}
```


