# **Reviewed approaches**:
- Saturation Loss + PAN Method
- GP-GAN (Gaussian-Poisson GAN)
- Deep Image Blending
- Poisson Blending
- V-LASIK’s Diffusion-Based Blending
- Alpha Blending (most basic)
- Multi-band Blending (Laplacian Pyramids)

## Saturation Loss + 2-stage PAN Method
Is considered the mmost realistic

- https://ar5iv.labs.arxiv.org/html/2306.05382


## GP-GAN (Gaussian-Poisson GAN)
Handles edges by GAN + gradient fusion (combines GANs with gradient-domain optimization)

- https://arxiv.org/pdf/1703.07195

## V-LASIK’s Diffusion-Based Blending

Combines ControlNet inpainting with cross-frame attention and latent-space blending. While designed for video, its frame-wise blending uses Inside-Out Normalization (ION) to align color/lighting statistics between the glasses and background

## Deep Image Blending
Integrates gradient-domain consistency (with Poisson) with style/texture losses from deep networks

- https://www.cs.toronto.edu/~lindell/teaching/2529/past_projects/2024/report/shivanshi-gupta.pdf

- https://ar5iv.labs.arxiv.org/html/2306.05382


## Poisson Blending
Handles edges by matching gradients

Eliminates sharp edges by enforcing gradient consistency across boundaries

Proven effectiveness in seamless cloning tasks

- https://www.ipol.im/pub/art/2016/163/article_lr.pdf
- https://www.cs.toronto.edu/~lindell/teaching/2529/past_projects/2024/report/shivanshi-gupta.pdf


# **Initial plan**:
1. Generate initial blending
2. Augment it
3. Assess versus some baseline (i.e. Poisson/Alpha blending)
4. Assess by training segmentation models on the synthetic dataset
