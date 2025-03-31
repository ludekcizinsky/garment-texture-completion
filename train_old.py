#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for TextureCompletion."""

import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from load_data import DatasetInpainting

logger = get_logger(__name__, log_level="INFO")

WANDB_TABLE_COL_NAMES = ["partial_image", "full_diffuse_img", "full_normal_img", "full_roughness_img", "prompt"]

def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.permute(0, 2, 3, 1)
    return image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for TextureCompletion.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="nn/material_gen",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="nn/material_gen",
        required=True,
        help="Path to pretrained vaes.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default="", help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="texture-inpainting-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether to debug",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training TextureCompletion. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='outputs/gen_textures_fl32_step100_1k_mix',
        help=(
            "dataset path"
        ),
    )
    parser.add_argument("--ddim_loss", action="store_true", help="Include DDIM loss in the training process.")
    

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def get_local_image(url):
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():


    
    # DataLoaders creation:
    train_dataset = DatasetInpainting(args.dataset_path, res=args.resolution, mask_ratio=0.5, is_train=True, debug=args.debug)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Train!
    global_step = 0
    first_epoch = 0

    # Resume from a previous checkpoint
    # TODO: add this


    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(batch["full_diffuse_img"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                # caption_input = tokenizer(list(batch["prompt"]), max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                # captions = caption_input.input_ids.to(latents.device)
                captions = tokenize_captions(list(batch["prompt"])).to(accelerator.device)
                encoder_hidden_states = text_encoder(captions)[0] # None #

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                partial_image_embeds = vae.encode(batch["partial_img"].to(weight_dtype)).latent_dist.mode()

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states) # None # 

                    # Sample masks for the original images.
                    image_mask_dtype = partial_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    partial_image_embeds = image_mask * partial_image_embeds

                # Concatenate the `partial_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, partial_image_embeds], dim=1) # [B, 8, H, W]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute the primary loss
                primary_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Compute the one-step DDIM target and loss
                if args.ddim_loss:
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
                    alpha_t = alphas_cumprod[timesteps]  # Alpha_t at current timestep
                    
                    sqrt_alpha_t = alpha_t.sqrt().view(-1, 1, 1, 1)
                    sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt().view(-1, 1, 1, 1)

                    # DDIM reverse step: estimate the noise-free latents at t=0
                    ddim_pred = (
                        noisy_latents  # Current latent scaled by next alpha  
                        - sqrt_one_minus_alpha_t * model_pred  # Remove noise proportional to the next timestep
                    ) / sqrt_alpha_t
                    
                    ddim_loss = F.mse_loss(ddim_pred, latents, reduction="mean")
                    
                    # Combine the losses with a weighting factor (adjust `lambda_ddim` as needed)
                    lambda_ddim = 0.5  # Adjust this weight based on your experiments
                    loss = primary_loss + lambda_ddim * ddim_loss
                else:
                    loss = primary_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


        # --------------- Validation logic
        if accelerator.is_main_process:
            if (
                (args.val_image_url is not None)
                and (epoch % args.validation_epochs == 0)
            ):
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with partial images."
                )
                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    vae=accelerator.unwrap_model(vae),
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                partial_image = get_local_image(args.val_image_url) # download_image(args.val_image_url)
                
                latents = []
                with torch.autocast(
                    str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                ):
                    for _ in range(args.num_validation_images):
                        latents.append(
                            pipeline(
                                args.validation_prompt,
                                image=partial_image,
                                num_inference_steps=20,
                                image_guidance_scale=1.5,
                                guidance_scale=7,
                                generator=generator,
                                output_type="latent",
                                return_dict=True,
                            )[0]
                        )

                # decode the latents and save the color, normal, and roughness images
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                        for latent in latents:
                            pt = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
                            diffuse = pipeline.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
                            pt = vae_normal.decode(latent / vae_normal.config.scaling_factor, return_dict=False)[0]
                            normal = pipeline.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
                            pt = vae_roughness.decode(latent / vae_roughness.config.scaling_factor, return_dict=False)[0]
                            roughness = pipeline.image_processor.postprocess(pt, output_type="pil", do_denormalize=[True])[0]
                            
                            diffuse.save("gen_texture_diffuse.png")
                            normal.save("gen_texture_normal.png")
                            roughness.save("gen_texture_roughness.png")
                            
                            wandb_table.add_data(
                                wandb.Image(partial_image), wandb.Image(diffuse), wandb.Image(normal), wandb.Image(roughness), args.validation_prompt
                            )
                        tracker.log({"validation": wandb_table})

                del pipeline
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
