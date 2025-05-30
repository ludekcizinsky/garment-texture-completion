import os

import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler
from helpers.utils import get_unet_model


class GarmentDenoiser(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.weight_dtype = torch.float16 if self.cfg.trainer.precision == "16-mixed" else torch.float32

        self._load_pretrained_components()
        self.unet = get_unet_model(cfg)
        self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod)

    def _get_encoded_prompt(self, prompt):

        # Tokenize (returns input_ids and attention_mask)
        text_inputs = self.tokenizer(
            [prompt],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt", 
        )
        text_inputs = {k: v.to(self.text_encoder.device) for k,v in text_inputs.items()}

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

        # Load and freeze the text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.model.diffusion_path, subfolder="tokenizer", 
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.model.diffusion_path, subfolder="text_encoder", 
            torch_dtype=self.weight_dtype 
        )
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Freeze the encoder/decoder 
        # VAE (obtained from DressCode)
        self.vae_diffuse = AutoencoderKL.from_pretrained(
            os.path.join(self.cfg.model.vae_path, "refine_vae"),
            subfolder="vae_checkpoint_diffuse",
            revision="fp32",
            local_files_only=True,
            torch_dtype=self.weight_dtype,
        )

        for module in [self.vae_diffuse]:
            for param in module.parameters():
                param.requires_grad = False
        
    def _classifier_free_guidance(self, text_embeds, null_prompt_embeds, partial_image_embeds):
        """
        # Conditioning dropout to support classifier-free guidance during inference. For more details
        # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
        """

        bsz = text_embeds.shape[0]
        dropout_prob = self.cfg.model.conditioning_dropout_prob

        if dropout_prob > 0:
            # Text
            random_p = torch.rand(bsz, device=text_embeds.device)
            prompt_mask = random_p < 2 * dropout_prob
            prompt_mask = prompt_mask.reshape(bsz, 1, 1)
            null_conditioning = null_prompt_embeds.repeat(bsz, 1, 1)
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

    def encode_latents(self, full_diffuse_img):
        with torch.no_grad():
            # returns z ∼ q(z|x) × scaling_factor
            latents = self.vae_diffuse.encode(full_diffuse_img.to(self.weight_dtype)).latent_dist.sample()
            return latents * self.vae_diffuse.config.scaling_factor
    
    def decode_latents(self, latents):
        with torch.no_grad():
            return self.vae_diffuse.decode(latents / self.vae_diffuse.config.scaling_factor).sample

    def encode_partial(self, partial_img):
        with torch.no_grad():
            # returns the deterministic encoding of the partial image
            return self.vae_diffuse.encode(partial_img.to(self.weight_dtype)).latent_dist.mode()

    def denoise_step(self, noisy_latents, timesteps, text_embeds):
        return self.unet(noisy_latents, timesteps, text_embeds).sample