import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from helpers.model import GarmentDenoiser
from helpers.losses import ddim_loss as ddim_loss_f


class GarmentInpainterModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.model = GarmentDenoiser(cfg)

        self.prompt = "fill the missing parts of a fabric texture matching the existing colors and style"
        self.null_prompt = ""

    def setup(self, stage=None):
        # Called once per device
        self.prompt_embeds = self.model._get_encoded_prompt(self.prompt).to(self.device, dtype=self._amp_dtype())
        self.null_prompt_embeds = self.model._get_encoded_prompt(self.null_prompt).to(self.device, dtype=self._amp_dtype())

    def _amp_dtype(self):
        return torch.float16 if self.trainer.precision == "16-mixed" else torch.float32


    def forward(self, noisy_latents, timesteps, text_embeds, partial_image_embeds):
        model_pred =  self.model.denoise_step(
                    torch.cat([noisy_latents, partial_image_embeds], dim=1),
                    timesteps,
                    text_embeds
                )
        return model_pred

    def _shared_step(self, batch):

        full_diffuse_imgs, partial_diffuse_imgs = batch["full_diffuse_img"], batch["partial_diffuse_img"]
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
        text_embeds = self.prompt_embeds.repeat(bsz, 1, 1)
        partial_image_embeds = self.model.encode_partial(partial_diffuse_imgs)
        if self.hparams.model.conditioning_dropout_prob > 0:
            text_embeds, partial_image_embeds = self.model._classifier_free_guidance(
                text_embeds, self.null_prompt_embeds, partial_image_embeds
            )

        # Denoising
        model_pred = self.forward(noisy_latents, timesteps, text_embeds, partial_image_embeds)

        return latents, noisy_latents, timesteps, model_pred, target

    def compute_losses(self, latents, noisy_latents, timesteps, model_pred, target):

        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if self.hparams.model.ddim_loss:
            alpha_t = self.model.alphas_cumprod[timesteps]
            ddim_loss = ddim_loss_f(alpha_t, latents, noisy_latents, model_pred)
            loss = mse_loss + 0.5 * ddim_loss
        else:
            loss = mse_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "ddim_loss": ddim_loss if self.hparams.model.ddim_loss else None,
        }

    def training_step(self, batch, batch_idx):

        latents, noisy_latents, timesteps, model_pred, target = self._shared_step(batch)
        losses = self.compute_losses(latents, noisy_latents, timesteps, model_pred, target)
        self.log_dict({f"train/{k}": v for k,v in losses.items() if v is not None},
                    on_step=True, on_epoch=False, prog_bar=True)
        self.log("step", self.global_step, prog_bar=False)

        
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        
        latents, noisy_latents, timesteps, model_pred, target = self._shared_step(batch)
        losses = self.compute_losses(latents, noisy_latents, timesteps, model_pred, target)
        self.log_dict({f"val/{k}": v for k,v in losses.items() if v is not None},
                    on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay,
        )
        
        return {
            "optimizer": optimizer,
        }
