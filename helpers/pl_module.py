import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from helpers.model import GarmentDenoiser
from helpers.losses import ddim_loss as ddim_loss_f


class GarmentInpainterModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = GarmentDenoiser(cfg)

    def training_step(self, batch, batch_idx):

        # Forward pass 
        full_diffuse_imgs, partial_diffuse_imgs = batch["full_diffuse_img"], batch["partial_diffuse_img"]
        latents, noisy_latents, timesteps, target, model_pred = self.model(full_diffuse_imgs, partial_diffuse_imgs)

        # Compute the loss
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("train/mse_loss", mse_loss, prog_bar=True, on_step=True, on_epoch=True)

        if self.hparams.model.ddim_loss:
            alphas_cumprod = self.model.noise_scheduler.alphas_cumprod.to(timesteps)
            alpha_t = alphas_cumprod[timesteps]  
            ddim_loss = ddim_loss_f(alpha_t, latents, noisy_latents, model_pred)
            self.log("train/ddim_loss", ddim_loss, prog_bar=True, on_step=True, on_epoch=True)
            loss = mse_loss + 0.5 * ddim_loss
        else:
            loss = mse_loss

        self.log("train/total_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        # Forward pass 
        full_diffuse_imgs, partial_diffuse_imgs = batch["full_diffuse_img"], batch["partial_diffuse_img"]
        latents, noisy_latents, timesteps, target, model_pred = self.model(full_diffuse_imgs, partial_diffuse_imgs)

        # Compute the loss
        mse_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("val/mse_loss", mse_loss, prog_bar=True, on_step=False, on_epoch=True)

        if self.hparams.model.ddim_loss:
            alphas_cumprod = self.model.noise_scheduler.alphas_cumprod.to(timesteps)
            alpha_t = alphas_cumprod[timesteps]  
            ddim_loss = ddim_loss_f(alpha_t, latents, noisy_latents, model_pred)
            self.log("val/ddim_loss", ddim_loss, prog_bar=True, on_step=False, on_epoch=True)
            loss = mse_loss + 0.5 * ddim_loss
        else:
            loss = mse_loss

        self.log("val/total_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.model.learning_rate,
            weight_decay=self.hparams.model.weight_decay,
        )
        
        return {
            "optimizer": optimizer,
        }
