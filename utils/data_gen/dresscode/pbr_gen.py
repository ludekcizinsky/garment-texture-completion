import torch
import diffusers
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import os

class DressCodePBRGen:
	def __init__(self):
		sd_device = "cuda"
		local_dir = "/workspace/garment-texture-completion/utils/data_gen/dresscode/material_gen"

		self.vae_diffuse = AutoencoderKL.from_pretrained(
			f"{local_dir}/refine_vae", subfolder="vae_checkpoint_diffuse", revision="fp16", local_files_only=True, torch_dtype=torch.float16).half().to(sd_device)

		self.vae_normal = AutoencoderKL.from_pretrained(
			f"{local_dir}/refine_vae", subfolder="vae_checkpoint_normal", revision="fp16", local_files_only=True, torch_dtype=torch.float16).half().to(sd_device)

		self.vae_roughness = AutoencoderKL.from_pretrained(
			f"{local_dir}/refine_vae", subfolder="vae_checkpoint_roughness", revision="fp16", local_files_only=True, torch_dtype=torch.float16).half().to(sd_device)

		self.invpipe = StableDiffusionPipeline.from_pretrained(f"{local_dir}", torch_dtype=torch.float16, safety_checker=None, vae=self.vae_diffuse)
		self.invpipe = self.invpipe.to(sd_device)

		def patch_conv(module):
			if isinstance(module, torch.nn.Conv2d):
				module.padding_mode="circular"

		self.invpipe.unet.apply(patch_conv)
		self.invpipe.vae.apply(patch_conv)
		self.vae_diffuse.apply(patch_conv)
		self.vae_normal.apply(patch_conv)
		self.vae_roughness.apply(patch_conv)

	def run(self, prompt, out_folder):

		with torch.no_grad():
			
			latents = self.invpipe([prompt], 512, 512, output_type = "latent", return_dict=True)[0]

			pt = self.vae_diffuse.decode(latents / self.vae_diffuse.config.scaling_factor, return_dict=False)[0]
			diffuse = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
			diffuse.save(os.path.join(out_folder, f"texture_diffuse.png"))

			pt = self.vae_normal.decode(latents / self.vae_normal.config.scaling_factor, return_dict=False)[0]
			normal = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
			normal.save(os.path.join(out_folder, f"texture_normal.png"))

			pt = self.vae_roughness.decode(latents / self.vae_roughness.config.scaling_factor, return_dict=False)[0]
			roughness = self.invpipe.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
			roughness.save(os.path.join(out_folder, f"texture_roughness.png"))


def download_models_locally():
	from huggingface_hub import snapshot_download

	# Define the repository ID
	repo_id = "IHe-KaiI/DressCode"

	# Specify the local directory where you want to download the files
	local_dir = "/workspace/garment-texture-completion/utils/data_gen/dresscode/material_gen"

	# Download only the 'material_gen' directory
	snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns="material_gen/*")


if __name__ == "__main__":

	# download_models_locally()
	generator = DressCodePBRGen()
	prompt = "green fabric with a floral pattern"
	out_folder = "/workspace/garment-texture-completion/utils/data_gen/dresscode/output"
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	
	generator.run(prompt, out_folder)