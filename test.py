import os
import torch
import lpips
import numpy as np
from PIL import Image
from transformers import CLIPTextModel
from diffusers import StableDiffusionInstructPix2PixPipeline, AutoencoderKL, UNet2DConditionModel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms

# Load LPIPS model (AlexNet is used by default)
lpips_model = lpips.LPIPS(net='alex').to("cuda")

def process_image(image, res):
    image = image.resize((res, res), Image.Resampling.NEAREST)
    image = np.array(image).astype(np.float32) / 255  # [0, 1]
    image = 2 * image - 1  # [-1, 1]
    return image

# Function to calculate SSIM, LPIPS, and PSNR
def calculate_metrics(generated_img, gt_img):
    """
    Calculates SSIM, LPIPS, and PSNR between generated and ground truth images.
    
    Args:
        generated_img (PIL.Image.Image): Generated image.
        gt_img (PIL.Image.Image): Ground truth image.
    
    Returns:
        dict: Metrics including SSIM, LPIPS, and PSNR.
    """
    # Convert images to NumPy arrays
    generated_np = np.array(generated_img)
    gt_np = np.array(gt_img)

    # Calculate SSIM
    if len(generated_np.shape) == 3:  # RGB images
        ssim_value = np.mean([
            ssim(generated_np[..., c], gt_np[..., c], data_range=255)
            for c in range(generated_np.shape[2])
        ])
    else:  # Grayscale images
        ssim_value = ssim(generated_np, gt_np, data_range=255)

    # Calculate PSNR
    psnr_value = psnr(gt_np, generated_np, data_range=255)

    # Convert to tensors for LPIPS
    transform = transforms.ToTensor()
    generated_tensor = transform(generated_img).unsqueeze(0).to("cuda")
    gt_tensor = transform(gt_img).unsqueeze(0).to("cuda")

    # Calculate LPIPS
    lpips_value = lpips_model(generated_tensor, gt_tensor).item()

    return {"SSIM": ssim_value, "LPIPS": lpips_value, "PSNR": psnr_value}

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

    def run(self, partual_img, save_to):
        with torch.no_grad():
            # Generate latent representation
            latents = self.invpipe(
                "fill the missing parts of a fabric texture matching the existing colors and style",
                image=partual_img,
                num_inference_steps=50,
                image_guidance_scale=1.5,
                guidance_scale=7,
                output_type="latent",
                return_dict=True,
            )[0] # torch.Size([1, 4, 128, 128])
            
            # Decode and save diffuse texture
            pt = self.vae_diffuse.decode(latents / self.vae_diffuse.config.scaling_factor, return_dict=False)[0]
            diffuse = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
            diffuse.save(os.path.join(save_to, "texture_diffuse.png"))
            
            # Decode and save normal map
            pt = self.vae_normal.decode(latents / self.vae_normal.config.scaling_factor, return_dict=False)[0]
            normal = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
            normal.save(os.path.join(save_to, "texture_normal.png"))

            # Decode and save roughness map
            pt = self.vae_roughness.decode(latents / self.vae_roughness.config.scaling_factor, return_dict=False)[0]
            roughness = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
            roughness.save(os.path.join(save_to, "texture_roughness.png"))

if __name__ == "__main__":
    res = 512
    unet_path = "checkpoints/completion_diffusion"
    pretrained_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    vae_path = "checkpoints"
    test_path = "datasets/testset"
    file_paths = sorted([os.path.join(test_path, f) for f in os.listdir(test_path)])
    print(f"Testing on {len(file_paths)} images")

    # Initialize generator
    gen = InpaintTexture(unet_path, pretrained_model, vae_path)

    # Define output and input paths
    output_folder = os.path.join("test_outputs", *unet_path.split("/")[-2:])

    # Create output directory if not exists
    if output_folder is not None and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Store metrics for all images
    results = []
    metrics_summary = {"diffuse": {"SSIM": 0, "LPIPS": 0, "PSNR": 0},
                       "normal": {"SSIM": 0, "LPIPS": 0, "PSNR": 0},
                       "roughness": {"SSIM": 0, "LPIPS": 0, "PSNR": 0},
                       "count": {"diffuse": 0, "normal": 0, "roughness": 0}}

    # Generate textures and calculate metrics for each partial image
    for idx, file_path in enumerate(file_paths):
        save_to = os.path.join(output_folder, file_path.split('/')[-2] + '_' + file_path.split('/')[-1])
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        
        # Read ground truth and partial images
        diffuse_path = os.path.join(file_path, "color.png")
        gt_diffuse = Image.open(diffuse_path).convert("RGB")
        gt_diffuse = gt_diffuse.resize((res, res), Image.Resampling.NEAREST)
        gt_diffuse.save(os.path.join(save_to, "gt_diffuse.png"))

        # Read optional ground truth normal and roughness images
        gt_normal_path = os.path.join(file_path, "normal.png")
        gt_roughness_path = os.path.join(file_path, "roughness.png")
        gt_normal = Image.open(gt_normal_path).convert("RGB") if os.path.exists(gt_normal_path) else None
        if gt_normal is not None:
            gt_normal = gt_normal.resize((res, res), Image.Resampling.NEAREST)
        gt_roughness = Image.open(gt_roughness_path).convert("RGB") if os.path.exists(gt_roughness_path) else None
        if gt_roughness is not None:
            gt_roughness = gt_roughness.resize((res, res), Image.Resampling.NEAREST)

        # Read partial image
        partual_img = Image.open(os.path.join(file_path, f"partual_color.png")).convert("RGB")
        partual_img.save(os.path.join(save_to, "input.png"))

        # Process partial image and run generator
        partual_img = process_image(partual_img, res)
        mask = Image.open(os.path.join(file_path, f"mask.png")).convert("L")
        mask = np.array(mask).astype(bool)
        Image.fromarray((mask.squeeze() * 255).astype(np.uint8)).save(os.path.join(save_to, "mask.png"))
        mask = 2 * mask - 1  # [-1, 1]

        # Run texture completion
        gen.run(partual_img, save_to)

        # Load generated images
        generated_diffuse = Image.open(os.path.join(save_to, "texture_diffuse.png")).convert("RGB")
        generated_diffuse = generated_diffuse.resize((res, res), Image.Resampling.NEAREST)
        generated_normal = Image.open(os.path.join(save_to, "texture_normal.png")).convert("RGB")
        generated_normal = generated_normal.resize((res, res), Image.Resampling.NEAREST)
        generated_roughness = Image.open(os.path.join(save_to, "texture_roughness.png")).convert("RGB")
        generated_roughness = generated_roughness.resize((res, res), Image.Resampling.NEAREST)

        # Calculate metrics for each property
        metrics = {
            "index": idx,
            "diffuse": calculate_metrics(generated_diffuse, gt_diffuse),
        }

        # Update metrics for diffuse
        metrics_summary["diffuse"]["SSIM"] += metrics["diffuse"]["SSIM"]
        metrics_summary["diffuse"]["LPIPS"] += metrics["diffuse"]["LPIPS"]
        metrics_summary["diffuse"]["PSNR"] += metrics["diffuse"]["PSNR"]
        metrics_summary["count"]["diffuse"] += 1

        # Update metrics for normal if available
        if gt_normal:
            metrics["normal"] = calculate_metrics(generated_normal, gt_normal)
            metrics_summary["normal"]["SSIM"] += metrics["normal"]["SSIM"]
            metrics_summary["normal"]["LPIPS"] += metrics["normal"]["LPIPS"]
            metrics_summary["normal"]["PSNR"] += metrics["normal"]["PSNR"]
            metrics_summary["count"]["normal"] += 1

        # Update metrics for roughness if available
        if gt_roughness:
            metrics["roughness"] = calculate_metrics(generated_roughness, gt_roughness)
            metrics_summary["roughness"]["SSIM"] += metrics["roughness"]["SSIM"]
            metrics_summary["roughness"]["LPIPS"] += metrics["roughness"]["LPIPS"]
            metrics_summary["roughness"]["PSNR"] += metrics["roughness"]["PSNR"]
            metrics_summary["count"]["roughness"] += 1

        results.append(metrics)
        print(f"Metrics for {file_path}: {metrics}")

    # Calculate average metrics
    avg_metrics = {
        "diffuse": {k: v / metrics_summary["count"]["diffuse"] for k, v in metrics_summary["diffuse"].items()},
        "normal": {k: v / metrics_summary["count"]["normal"] if metrics_summary["count"]["normal"] > 0 else None for k, v in metrics_summary["normal"].items()},
        "roughness": {k: v / metrics_summary["count"]["roughness"] if metrics_summary["count"]["roughness"] > 0 else None for k, v in metrics_summary["roughness"].items()},
    }

    # Save results to file
    results_path = os.path.join(output_folder, "metrics.json")
    with open(results_path, "w") as f:
        import json
        json.dump({"per_image": results, "average": avg_metrics}, f, indent=4)
    print(f"Metrics saved to {results_path}")

    # Print average metrics
    print("Average Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key.capitalize()} - SSIM: {value['SSIM']:.4f}, LPIPS: {value['LPIPS']:.4f}, PSNR: {value['PSNR']:.4f}")