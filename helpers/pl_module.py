import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from helpers.model import GarmentDenoiser
from helpers.losses import ddim_loss as ddim_loss_f
from helpers.metrics import compute_all_metrics
from helpers.data_utils import denormalise_image_torch, pil_to_tensor
from helpers.plots import get_input_output_plot
from helpers.utils import get_optimizer, get_lr_scheduler

from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionInpaintPipeline

import wandb

class GarmentInpainterModule(pl.LightningModule):
    def __init__(self, cfg, trn_dataloader):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.lr = cfg.optim.lr
        self.weight_decay = cfg.optim.weight_decay
        self.model = GarmentDenoiser(cfg)

        self.trn_dataloader = trn_dataloader
        self.val_results = []

        self.prompt = "fill the missing parts of a fabric texture matching the existing colors and style"
        self.null_prompt = ""

        self._get_inference_pipe()



    def _get_inference_pipe(self):
        if self.hparams.model.is_inpainting:
            self.inference_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.cfg.model.diffusion_path,
                unet=self.model.unet,
                vae=self.model.vae_diffuse,
                revision=None,
                requires_safety_checker=False,
                safety_checker=None,
                torch_dtype=torch.float32,
            ).to("cuda")           
        else:
            self.inference_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.cfg.model.diffusion_path,
                unet=self.model.unet,
                vae=self.model.vae_diffuse,
                revision=None,
                requires_safety_checker=False,
                safety_checker=None,
                torch_dtype=torch.float32,
            ).to("cuda")           

    def setup(self, stage=None):
        # Called once per device
        self.prompt_embeds = self.model._get_encoded_prompt(self.prompt).to(self.device, dtype=self._amp_dtype())
        self.null_prompt_embeds = self.model._get_encoded_prompt(self.null_prompt).to(self.device, dtype=self._amp_dtype())

    def _amp_dtype(self):
        return torch.float16 if self.cfg.trainer.precision == "16-mixed" else torch.float32


    def forward(self, noisy_latents, timesteps, text_embeds, partial_image_embeds, masks):
        if self.hparams.model.is_inpainting:
            denoiser_input = torch.cat([noisy_latents, masks, partial_image_embeds], dim=1)
        else:
            denoiser_input = torch.cat([noisy_latents, partial_image_embeds], dim=1)

        model_pred =  self.model.denoise_step(
                    denoiser_input,
                    timesteps,
                    text_embeds
                )
        return model_pred

    def _shared_step(self, batch):

        full_diffuse_imgs, partial_diffuse_imgs, masks = batch["full_diffuse_img"], batch["partial_diffuse_img"], batch["mask"]
        bsz = full_diffuse_imgs.shape[0]

        # Noisy latents and timesteps
        latents = self.model.encode_latents(full_diffuse_imgs)

        noise = torch.randn_like(latents, device=latents.device)
        timesteps = torch.randint(
            0,
            self.model.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        noisy_latents = self.model.noise_scheduler.add_noise(latents, noise, timesteps)
        target = noise

        # Conditioning
        masks = torch.nn.functional.interpolate(masks, scale_factor=1/8)
        text_embeds = self.prompt_embeds.repeat(bsz, 1, 1)
        partial_image_embeds = self.model.encode_partial(partial_diffuse_imgs)
        if self.hparams.model.conditioning_dropout_prob > 0:
            text_embeds, partial_image_embeds = self.model._classifier_free_guidance(
                text_embeds, self.null_prompt_embeds, partial_image_embeds
            )

        # Denoising
        model_pred = self.forward(noisy_latents, timesteps, text_embeds, partial_image_embeds, masks)

        return latents, noisy_latents, timesteps, model_pred, target

    def compute_losses(self, latents, noisy_latents, timesteps, model_pred, target):

        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.hparams.optim.ddim_loss:
            alpha_t = self.model.alphas_cumprod[timesteps]
            ddim_loss = ddim_loss_f(alpha_t, latents, noisy_latents, model_pred)
            loss = mse_loss + self.hparams.optim.ddim_loss_weight * ddim_loss
        else:
            loss = mse_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "ddim_loss": ddim_loss if self.hparams.optim.ddim_loss else None,
        }

    def training_step(self, batch, batch_idx):

        latents, noisy_latents, timesteps, model_pred, target = self._shared_step(batch)
        losses = self.compute_losses(latents, noisy_latents, timesteps, model_pred, target)
        self.log_dict({f"train/{k}": v for k,v in losses.items() if v is not None}, on_step=True)
        self.log("step", self.global_step)
 
        return losses["loss"]

    def validation_step(self, batch, batch_idx):

        # Latent space metrics        
        latents, noisy_latents, timesteps, model_pred, target = self._shared_step(batch)
        losses = self.compute_losses(latents, noisy_latents, timesteps, model_pred, target)
        bsz = latents.shape[0]
        self.log_dict({f"val/{k}": v for k,v in losses.items() if v is not None}, on_step=False, on_epoch=True, batch_size=bsz)

        # Image space metrics
        reconstructed_imgs = self.inference(batch["partial_diffuse_img"], masks=batch["mask"])
        reconstructed_imgs_tensors = torch.stack([pil_to_tensor(img) for img in reconstructed_imgs]).to(self.device)
        target_imgs = denormalise_image_torch(batch["full_diffuse_img"])
        image_metrics = compute_all_metrics(reconstructed_imgs_tensors, target_imgs)
        self.val_results.append(image_metrics)

        # find index of the easiest and hardest sample based on ssim
        if batch_idx % 3 == 0:
            ssim_scores = image_metrics["ssim"]
            easiest_sample_idx = ssim_scores.argmax()
            hardest_sample_idx = ssim_scores.argmin()

            easiest_pred = reconstructed_imgs[easiest_sample_idx]
            hardest_pred = reconstructed_imgs[hardest_sample_idx]

            easiest_pred_plot = get_input_output_plot(
                batch["partial_diffuse_img"][easiest_sample_idx],
                batch["full_diffuse_img"][easiest_sample_idx],
                easiest_pred
            )

            hardest_pred_plot = get_input_output_plot(
                batch["partial_diffuse_img"][hardest_sample_idx],
                batch["full_diffuse_img"][hardest_sample_idx],
                hardest_pred
            )
            
            
            wandb.log(
                {
                    f"val-images/easiest_pred_batch_{batch_idx}": wandb.Image(easiest_pred_plot, caption=batch["name"][easiest_sample_idx]),
                    f"val-images/hardest_pred_batch_{batch_idx}": wandb.Image(hardest_pred_plot, caption=batch["name"][hardest_sample_idx]),
                },
                step=self.global_step
            )

        # log also selected texture names
        selected_texture_names = self.cfg.data.val_sel_texture_names
        batch_names = batch["name"]
        sel_figures = dict()
        for i, name in enumerate(selected_texture_names):
            if name in batch_names:
                idx = batch_names.index(name)
                figure = get_input_output_plot(
                    batch["partial_diffuse_img"][idx],
                    batch["full_diffuse_img"][idx],
                    reconstructed_imgs[idx]
                )
                sel_figures[f"val-images/sel_figure_{i}"] = wandb.Image(figure, caption=name)

        wandb.log(sel_figures, step=self.global_step)


    def on_validation_epoch_end(self):
        metrics = {}
        for output in self.val_results:
            for k, v in output.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

        agg_metrics = {}
        for k, v in metrics.items():
            all_results = torch.cat(v)
            agg_metrics[k] = all_results.mean()

        self.log_dict(
            {
                f"val/{k}": v for k, v in agg_metrics.items()
            }
        )
        self.val_results = []    

    def predict_step(self, batch, batch_idx):
        reconstructed_imgs = self.inference(batch["partial_diffuse_img"], masks=batch["mask"])
        reconstructed_imgs_tensors = torch.stack([pil_to_tensor(img) for img in reconstructed_imgs]).to(self.device)
        target_imgs = denormalise_image_torch(batch["full_diffuse_img"])
        image_metrics = compute_all_metrics(reconstructed_imgs_tensors, target_imgs)

        return image_metrics

    def configure_optimizers(self):
        return_dict = {}
        return_dict = get_optimizer(self.cfg, self.model, return_dict)
        return_dict = get_lr_scheduler(self.cfg, return_dict)
        return return_dict

    def on_save_checkpoint(self, checkpoint):
        trn_dataset = self.trn_dataloader.dataset
        checkpoint['train_dataset_state'] = trn_dataset.state_dict()

    def on_load_checkpoint(self, checkpoint):
        trn_dataset = self.trn_dataloader.dataset
        trn_dataset.load_state_dict(checkpoint['train_dataset_state'])


    def inference(self,
                partial_diffuse_imgs,
                masks,
                num_inference_steps=50,
                guidance_scale=1.5,
                image_guidance_scale=5.0,
                strength=0.8):

        prompts = [self.prompt]*len(partial_diffuse_imgs)
        zero_one_img_tensors = denormalise_image_torch(partial_diffuse_imgs)
        self.inference_pipe.unet = self.model.unet.to(dtype=torch.float32)
        self.inference_pipe.vae = self.model.vae_diffuse.to(dtype=torch.float32)

        if self.hparams.model.is_inpainting:
            preds = self.inference_pipe(
                prompts,
                image=zero_one_img_tensors,
                num_inference_steps=num_inference_steps,
                mask_image=masks,
                guidance_scale=guidance_scale, # text guidance scale
                strength=strength
            ).images
        else:
            preds = self.inference_pipe(
                prompts,
                image=zero_one_img_tensors,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale,
            ).images

        return preds
