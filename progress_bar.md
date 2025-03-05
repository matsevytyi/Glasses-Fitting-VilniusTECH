## TODO:
- LaMa inpainting
(https://github.com/advimman/lama)
- Latent Diffusion Inpainting
(https://github.com/nickyisadog/latent-diffusion-inpainting?tab=readme-ov-file)
- CM-GAN inpainting
(https://github.com/htzheng/CM-GAN-Inpainting?tab=readme-ov-file#Code-for-On-the-fly-Object-aware-Mask-Generation)
- SmartBrush
(StableDiffusion)
(https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_SmartBrush_Text_and_Shape_Guided_Object_Inpainting_With_Diffusion_Model_CVPR_2023_paper.pdf)
- RePaint 
(StableDiffusion) 
(https://arxiv.org/pdf/2201.09865)
- Wasserstein Gan GP 
(in Pytorch)
- CoModGAN
- Clip-Guided Inpainting
- Deep Image Blending
- V-LASIK’s Diffusion-Based Blending
- Saturation Loss + 2-stage PAN Method
- HoloUNet
- Multi-stage Blendd Diffusion 
(https://replicate.com/arielreplicate/multi-stage-blended-diffusion)
- IP adapter
(plugged into Stable Diffusion to influence the result)
(https://huggingface.co/docs/diffusers/main/using-diffusers/ip_adapter)
- ControlNet
(plugged in Stable Diffusion to influence the result)
- solutions for Image Harmonization

## In progress:
- Poisson blending (working version) 
- Alpha blending (valid version) 

## Implemented:
- Stable Diffusion Impainting
- Latent Space Blending
- MultiBand Blending
(gaussian pyramid for mask, laplassian piramid for images, combine and then restore the image)

## Didn't work:
- GP-GAN 
(too old dependencies)
- Stable Diffusion General Model 
(doesn't work)
- 