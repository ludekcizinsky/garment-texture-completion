import os

import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

class GarmentDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self._load_pretrained_components()
        self._modify_unet()
        prompt = "fill the missing parts of a fabric texture matching the existing colors and style"
        self.register_buffer("prompt_embeds", self._get_encoded_prompt(prompt))
        self.register_buffer("null_prompt_embeds", self._get_encoded_prompt(""))

    def _get_encoded_prompt(self, prompt):

        # Tokenize (returns input_ids and attention_mask)
        text_inputs = self.tokenizer(
            [prompt],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt", 
        )
        text_inputs = {k: v.to("cuda") for k,v in text_inputs.items()}


        # Encode with text encoder
        with torch.no_grad():  # Optional, since you're not training the encoder
            text_embeddings = self.text_encoder(**text_inputs).last_hidden_state  # shape: (1, 77, 768) for CLIP ViT-B/32
        
        return text_embeddings

    def _load_pretrained_components(self):
        """
        Load pretrained components from the specified paths.
        Args:
            cfg: Configuration object containing paths and model names.
        """

        # SD
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.cfg.model.diffusion_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.model.diffusion_path, subfolder="tokenizer", 
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.cfg.model.diffusion_path, subfolder="unet", 
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.model.diffusion_path, subfolder="text_encoder", 
        )
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # VAE (obtained from DressCode)
        self.vae_diffuse = AutoencoderKL.from_pretrained(
            os.path.join(self.cfg.vae_path, "refine_vae"),
            subfolder="vae_checkpoint_diffuse",
            revision="fp32",
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self.vae_normal = AutoencoderKL.from_pretrained(
            os.path.join(self.cfg.vae_path, "refine_vae"),
            subfolder="vae_checkpoint_normal",
            revision="fp32",
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        self.vae_roughness = AutoencoderKL.from_pretrained(
            os.path.join(self.cfg.vae_path, "refine_vae"),
            subfolder="vae_checkpoint_roughness",
            revision="fp32",
            local_files_only=True,
            torch_dtype=torch.float32,
        )

        for module in [self.vae_diffuse, self.vae_normal, self.vae_roughness]:
            for param in module.parameters():
                param.requires_grad = False
        
    def _modify_unet(self):
        """
        TextureCompletion extends the input channels of the original UNet to take condition images.
        This method modifies the UNet's input convolution layer.
        """

        # Set new number of input channels. For example, extending from 4 to 8.
        in_channels = 8
        out_channels = self.unet.conv_in.out_channels
        
        # Update the model configuration (if used later for saving or further adjustments).
        self.unet.register_to_config(in_channels=in_channels)
        
        # Create a new conv layer with the extended number of input channels.
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels,
                out_channels,
                self.unet.conv_in.kernel_size,
                self.unet.conv_in.stride,
                self.unet.conv_in.padding
            )
            # Initialize the new weights: copy weights for the original 4 channels and zero the rest.
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in

    def _classifier_free_guidance(self, text_embeds, partial_image_embeds):
        """
        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        """

        bsz = text_embeds.shape[0]
        dropout_prob = self.cfg.model.conditioning_dropout_prob

        if dropout_prob > 0:
            # Text
            random_p = torch.rand(bsz)
            prompt_mask = random_p < 2 * dropout_prob
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            null_conditioning = self.null_prompt_embeds.repeat(bsz, 1, 1)
            text_embeds = torch.where(prompt_mask, null_conditioning, text_embeds)

            # Partial image
            image_mask_dtype = partial_image_embeds.dtype
            image_mask = 1 - (
                (random_p >= dropout_prob).to(image_mask_dtype)
                * (random_p < 3 * dropout_prob).to(image_mask_dtype)
            )
            image_mask = image_mask.reshape(bsz, 1, 1, 1)
            partial_image_embeds = image_mask * partial_image_embeds
        
        return text_embeds, partial_image_embeds


    def forward(self, full_diffuse_img, partial_img):

        bsz = full_diffuse_img.shape[0]

        # Noisy input
        latents = self.vae_diffuse.encode(full_diffuse_img).latent_dist.sample()
        latents = latents * self.vae_diffuse.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,)).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Conditioning
        text_embeds = self.prompt_embeds.repeat(bsz, 1, 1) 
        partial_image_embeds = self.vae_diffuse.encode(partial_img).latent_dist.mode()

        if self.cfg.model.conditioning_dropout_prob > 0:
            text_embeds, partial_image_embeds = self._classifier_free_guidance(
                text_embeds, partial_image_embeds
            )

        concatenated_noisy_latents = torch.cat([noisy_latents, partial_image_embeds], dim=1)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")


        # Denoising
        model_pred = self.unet(concatenated_noisy_latents, timesteps, text_embeds).sample

        return latents, noisy_latents, timesteps, target, model_pred
