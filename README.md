# MSAN-Pattern Recognition-2025

 MSAN: Multiscale self-attention network for pansharpening
 
Paper link: https://www.sciencedirect.com/science/article/abs/pii/S0031320325001013

The project operates under the framework of PyTorch.

For training, please run the train.py file. For testing, use the test_RS.py file for reduced-scale images and the test_lu_full.py file for full-scale images. If the images are too large, you can use the test_lu_full_crop.py file instead.

We would be pleased if you can cite this paper, and please refer to:

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
  keywords = {Multiscale,Pansharpening,Self-attention,Swin Transformer},

}


