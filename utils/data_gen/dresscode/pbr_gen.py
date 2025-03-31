import torch
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL

from tqdm import tqdm
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
		self.invpipe.set_progress_bar_config(disable=True)

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

def read_txt(path):
	with open(path, "r") as f:
		lines = f.readlines()
		lines = [line.strip() for line in lines]		
	return lines

if __name__ == "__main__":

	# Read prompt data
	project_path = "/workspace/garment-texture-completion"
	prompt_data = list()
	for file_name in ["colours.txt", "patterns.txt", "materials.txt"]:
		prompt_data.append(read_txt(os.path.join(project_path, "utils/data_gen/dresscode/queries", file_name)))

	# Create a combination of the three lists
	prompts = []
	for colour in prompt_data[0]:
		for pattern in prompt_data[1]:
			for material in prompt_data[2]:
				prompts.append(f"{material} {colour} {pattern}")

	# Generate the PBR textures
	generator = DressCodePBRGen()
	out_folder = f"{project_path}/utils/data_gen/dresscode/output"
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)

	for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):

		# create a subfolder for the output based on the prompt
		subfolder_name = prompt.replace(" ", "_")
		os.makedirs(os.path.join(out_folder, subfolder_name), exist_ok=True)
		subfolder_path = os.path.join(out_folder, subfolder_name)

		# Generate the textures
		generator.run(prompt, subfolder_path)