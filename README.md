[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

# Project 2

**Authors**: Jiachen Lu (jiachen.lu@epfl.ch), Yingxuan You (yingxuan.you@epfl.ch), Shunchang Liu (shunchang.liu@epfl.ch)

## Setup

To set up the project environment, follow these steps:

1. **Install the dependencies**:
```bash
git clone https://github.com/CS-433/ml-project-2-pbr.git
cd ml-project-2-pbr

conda env create -f environment.yaml
conda activate pbr
```
2. **Download the pre-trained models**:
Download the [PBR texture VAEs](https://huggingface.co/IHe-KaiI/DressCode/tree/main/material_gen) pre-trained by [DressCode](https://github.com/IHe-KaiI/DressCode) to ```./checkpoints``` folder.
Download our pre-trained texture completion diffusion model([completion_diffusion.zip](https://1drv.ms/f/c/d70f26d613e83858/ErrnQtxxg6ZKjIDdLQYJu6cBMCjvf1ZvNnacwrhAE-S3UQ?e=YBaKl4)) and unzip it at ```./checkpoints``` folder.
The ```./checkpoints``` directory structure should follow the below hierarchy:
```
${Project}  
|-- checkpoints  
|   |-- model_index.json
|   |-- completion_diffusion
|   |   |-- feature_extractor
|   |   |   |-- preprocessor_config.json
|   |   |-- safety_checker
|   |   |   |-- config.json
|   |   |   |-- model.safetensors
|   |   |-- scheduler
|   |   |   |-- scheduler_config.json
|   |   |-- text_encoder
|   |   |   |-- config.json
|   |   |   |-- model.safetensors
|   |   |-- tokenizer
|   |   |   |-- merges.txt
|   |   |   |-- special_tokens_map.json
|   |   |   |-- tokenizer_config.json
|   |   |   |-- vocab.json
|   |   |-- unet
|   |   |   |-- config.json
|   |   |   |-- diffusion_pytorch_model.safetensors
|   |-- refine_vae
|   |   |-- vae_checkpoint_diffuse
|   |   |   |-- diffusion_pytorch_model.safetensors
|   |   |-- vae_checkpoint_normal
|   |   |   |-- diffusion_pytorch_model.safetensors
|   |   |-- vae_checkpoint_roughness
|   |   |   |-- diffusion_pytorch_model.safetensors
```

## Quick Demo
Run our pre-trained texture completion diffusion model using sample in examples:
```bash
python demo.py --partial_img examples/partial_color.png --mask examples/mask.png
```
The output is at `./outputs` folder.

## Test
1. **Download the test set**:
Download the test set with **500** paired patrial-complete PBR texture maps from [OneDrive](https://1drv.ms/f/c/d70f26d613e83858/ErrnQtxxg6ZKjIDdLQYJu6cBMCjvf1ZvNnacwrhAE-S3UQ?e=nd1okN) ```testset.zip``` and unzip at ```./datasets``` folder.
2. **Evaluate the performance**:
```bash
python test.py
```
You will get the results on SSIM, LPIPS, PSNR:
| Model                  | SSIM ↑ |  LPIPS ↓ |  PSNR ↑ | Log | 
|------------------------|--------|---------|----------|--------|
| Texture completion diffusion      | 0.488   | 0.335 | 18.22 | [log](logs/metrics.json) |

## Train
1. **Download the training set**:
Download the training set with **27k** paired patrial-complete PBR texture maps [OneDrive](https://1drv.ms/f/c/d70f26d613e83858/ErrnQtxxg6ZKjIDdLQYJu6cBMCjvf1ZvNnacwrhAE-S3UQ?e=nd1okN) ```fabric_w_logo.zip``` and unzip at ```./datasets``` folder.
2. **Training**:
```bash
bash train.sh
```

## Acknowledgement
This repo is extended from the excellent work [diffusers](https://github.com/huggingface/diffusers), [InstructPix2Pix](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pix2pix), [DressCode](https://github.com/IHe-KaiI/DressCode/tree/main). We thank the authors for releasing the codes.