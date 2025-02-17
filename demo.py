import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from transformers import CLIPTextModel
from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel

def process_image(image, res):
    image = image.resize((res, res), Image.Resampling.NEAREST)
    image = np.array(image).astype(np.float32) / 255  # [0, 1]
    image = 2 * image - 1  # [-1, 1]
    return image

class InpaintTexture:
    def __init__(self, unet_path, pretrained_model, vae_path):
        sd_device = "cuda"

        # Load VAE models for different texture properties
        self.vae_diffuse = AutoencoderKL.from_pretrained(
            vae_path + "/refine_vae",
            subfolder="vae_checkpoint_diffuse",
            revision="fp32",
            local_files_only=True,
            torch_dtype=torch.float32
        ).to(sd_device)

        self.vae_normal = AutoencoderKL.from_pretrained(
            vae_path + "/refine_vae",
            subfolder="vae_checkpoint_normal",
            revision="fp32",
            local_files_only=True,
            torch_dtype=torch.float32
        ).to(sd_device)

        self.vae_roughness = AutoencoderKL.from_pretrained(
            vae_path + "/refine_vae",
            subfolder="vae_checkpoint_roughness",
            revision="fp32",
            local_files_only=True,
            torch_dtype=torch.float32
        ).to(sd_device)
        
        print('Loading unet ...')
        unet = UNet2DConditionModel.from_pretrained(
                unet_path, subfolder="unet", revision=None
            )
            
        print('Loading text_encoder ...')
        text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model, subfolder="text_encoder", revision=None
            )

        print('Loading pipline ...')
        self.invpipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                pretrained_model,
                unet=unet,
                text_encoder=text_encoder,
                vae=self.vae_diffuse,
                revision=None,
                safety_checker=None,
                torch_dtype=torch.float32
            ).to("cuda")
        generator = torch.Generator("cuda").manual_seed(0)

    def run(self, partial_img, output_folder):
        with torch.no_grad():
            # Generate latent representation
            latents = self.invpipe(
                "fill the missing parts of a fabric texture matching the existing colors and style",
                image=partial_img,
                num_inference_steps=20,
                image_guidance_scale=1.5, #1.5,
                guidance_scale=7,
                output_type="latent",
                return_dict=True,
            )[0]
            
            # Decode and save diffuse texture
            pt = self.vae_diffuse.decode(latents / self.vae_diffuse.config.scaling_factor, return_dict=False)[0]
            diffuse = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
            diffuse.save(os.path.join(output_folder, "texture_diffuse.png"))
            
            # Decode and save normal map
            pt = self.vae_normal.decode(latents / self.vae_normal.config.scaling_factor, return_dict=False)[0]
            normal = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
            normal.save(os.path.join(output_folder, "texture_normal.png"))

            # Decode and save roughness map
            pt = self.vae_roughness.decode(latents / self.vae_roughness.config.scaling_factor, return_dict=False)[0]
            roughness = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
            roughness.save(os.path.join(output_folder, "texture_roughness.png"))


if __name__ == "__main__":
    # Define command-line argument parser
    parser = argparse.ArgumentParser(description="Script for image processing with diffusion models.")
    
    # Add arguments with default values
    parser.add_argument("--res", type=int, default=512, help="Resolution of the image (default: 512)")
    parser.add_argument("--unet_path", type=str, default="checkpoints/completion_diffusion", help="Path to the UNet checkpoint directory (default: checkpoints/completion_diffusion)")
    parser.add_argument("--pretrained_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Path to the pretrained stable diffusion model (default: stable-diffusion-v1-5/stable-diffusion-v1-5)")
    parser.add_argument("--vae_path", type=str, default="checkpoints", help="Path to the VAE checkpoint directory (default: checkpoints)")
    parser.add_argument("--output_path", type=str, default="outputs", help="Output path for the generated textures (default: outputs)")
    parser.add_argument("--partial_img", type=str, required=True, help="Path to the partial color input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to the mask image")
    

    # Parse command-line arguments
    args = parser.parse_args()

    # Assign parsed arguments to variables
    res = args.res
    unet_path = args.unet_path
    pretrained_model = args.pretrained_model
    vae_path = args.vae_path
    partial_path = args.partial_img
    mask_path = args.mask

    # Initialize generator
    gen = InpaintTexture(unet_path, pretrained_model, vae_path)

    # Define output and input paths
    output_folder = os.path.join(args.output_path, f"res_{res}")

    # Create output directory if not exists
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load partial image    
    partial_img = Image.open(partial_path).convert("RGB")
    partial_img.save(os.path.join(output_folder, "input.png"))
    
    # Load mask
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask).astype(bool)
    Image.fromarray((mask.squeeze() * 255).astype(np.uint8)).save(os.path.join(output_folder, "mask.png"))
    
    # Process partial_img and mask
    partial_img = process_image(partial_img, res)
    mask = 2 * mask - 1
    
    # Run texture completion
    gen.run(partial_img, output_folder)