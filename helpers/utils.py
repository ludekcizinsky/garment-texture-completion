import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from peft import LoraConfig
from diffusers.training_utils import cast_training_params

from diffusers import UNet2DConditionModel
import torch.nn as nn

def get_optimizer(cfg, model, return_dict):

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optim.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.optim.weight_decay
    )

    return_dict["optimizer"] = optimizer

    return return_dict

def get_lr_scheduler(cfg, return_dict):

    optimizer = return_dict["optimizer"]

    if cfg.optim.use_cosine_scheduler:
        print("--- FYI: Using Cosine Scheduler")
        # Make your the two schedulers
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cfg.trainer.max_steps - cfg.optim.warmup_steps,
            eta_min=cfg.optim.min_lr
        )

        # Chain them *sequentially*, switching at warmup_steps
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.optim.warmup_steps]
        )

        return_dict["lr_scheduler"] = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

    return return_dict


def modify_unet(unet):
    """
    TextureCompletion extends the input channels of the original UNet to take condition images.
    This method modifies the UNet's input convolution layer.
    """

    # Set new number of input channels. For example, extending from 4 to 8.
    in_channels = 8
    out_channels = unet.conv_in.out_channels
    
    # Update the model configuration (if used later for saving or further adjustments).
    unet.register_to_config(in_channels=in_channels)
    
    # Create a new conv layer with the extended number of input channels.
    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels,
            out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding
        )
        # Initialize the new weights: copy weights for the original 4 channels and zero the rest.
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    return unet

def init_finetune_unet(cfg):
    print("--- FYI: Initializing Finetune UNet")

    unet = UNet2DConditionModel.from_pretrained(
        cfg.model.diffusion_path, subfolder="unet",
    )
    unet.enable_gradient_checkpointing()
    if not cfg.model.is_inpainting:
        unet = modify_unet(unet)

    return unet

def init_custom_unet(cfg):
    print("--- FYI: Initializing Custom UNet")

    model_id = cfg.model.diffusion_path
    config = UNet2DConditionModel.load_config(
        model_id, subfolder="unet"
    )

    config['cross_attention_dim'] = cfg.model.cross_attention_dim
    config['block_out_channels'] = cfg.model.block_out_channels

    unet = UNet2DConditionModel.from_config(config)
    unet.enable_gradient_checkpointing()

    return unet

def init_lora_unet(cfg):

    path = "timbrooks/instruct-pix2pix"
    print(f"--- FYI: Initializing Lora UNet from {path}")

    # load and freeze the unet
    unet = UNet2DConditionModel.from_pretrained(
        path, subfolder="unet",
    )
    for param in unet.parameters():
        param.requires_grad = False

    unet.enable_gradient_checkpointing()

    lora_unet_cfg = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    # unet = get_peft_model(unet, lora_unet_cfg)
    unet.add_adapter(lora_unet_cfg)
    if cfg.trainer.precision == "16-mixed":
        cast_training_params(unet, dtype=torch.float32)

    return unet

def init_pretrained_unet(cfg):
    """
    This should be used for the evaluation only.
    """
    print("--- FYI: Initializing Pretrained UNet")
    unet = UNet2DConditionModel.from_pretrained(
        cfg.model.diffusion_path, subfolder="unet",
    )

    return unet

def get_unet_model(cfg):
    
    if cfg.model.train_from_scratch:
        unet = init_custom_unet(cfg)
    elif cfg.model.train_with_lora:
        unet = init_lora_unet(cfg)
    elif cfg.model.use_pretrained_unet:
        unet = init_pretrained_unet(cfg)
    else:
        unet = init_finetune_unet(cfg)

    return unet