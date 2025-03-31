import os

import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel


class GarmentInpainterModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters() 
        self.load_pretrained_components()
        self.modify_unet()


    def load_pretrained_components(self):
        """
        Load pretrained components from the specified paths.
        Args:
            cfg: Configuration object containing paths and model names.
        """

        # SD
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer", 
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="unet", 
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder", 
        )

        if self.cfg.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

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
        


    def modify_unet(self):
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

    def setup(self, stage=None):
        self.generator = torch.Generator(device=self.device).manual_seed(self.hparams.seed)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.adam_weight_decay,
        )
        
        return {
            "optimizer": optimizer,
        }
