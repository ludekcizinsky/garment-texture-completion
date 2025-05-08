import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from helpers.model import GarmentDenoiser
from helpers.losses import ddim_loss as ddim_loss_f
from helpers.metrics import compute_all_metrics
from helpers.data_utils import denormalise_image_torch

from tqdm import tqdm

class GarmentInpainterModule(pl.LightningModule):
    def __init__(self, cfg, trn_dataloader):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.lr = cfg.optim.lr
        self.weight_decay = cfg.optim.weight_decay
        self.model = GarmentDenoiser(cfg)

        self.trn_dataloader = trn_dataloader

        self.prompt = "fill the missing parts of a fabric texture matching the existing colors and style"
        self.null_prompt = ""

    def setup(self, stage=None):
        # Called once per device
        self.prompt_embeds = self.model._get_encoded_prompt(self.prompt).to(self.device, dtype=self._amp_dtype())
        self.null_prompt_embeds = self.model._get_encoded_prompt(self.null_prompt).to(self.device, dtype=self._amp_dtype())

    def _amp_dtype(self):
        return torch.float16 if self.cfg.trainer.precision == "16-mixed" else torch.float32


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
        self.log_dict({f"train/{k}": v for k,v in losses.items() if v is not None}, on_step=True)
        self.log("step", self.global_step)
 
        return losses["loss"]

    def validation_step(self, batch, batch_idx):

        # Latent space metrics        
        latents, noisy_latents, timesteps, model_pred, target = self._shared_step(batch)
        losses = self.compute_losses(latents, noisy_latents, timesteps, model_pred, target)
        self.log_dict({f"val/{k}": v for k,v in losses.items() if v is not None}, on_step=False, on_epoch=True)

        # Image space metrics
        reconstructed_imgs = self.inference(batch["partial_diffuse_img"])
        reconstructed_imgs = denormalise_image_torch(reconstructed_imgs)
        target_imgs = denormalise_image_torch(batch["full_diffuse_img"])
        metrics = compute_all_metrics(reconstructed_imgs, target_imgs)
        self.log_dict({f"val/{k}": v for k,v in metrics.items()}, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        trn_dataset = self.trn_dataloader.dataset
        checkpoint['train_dataset_state'] = trn_dataset.state_dict()

    def on_load_checkpoint(self, checkpoint):
        trn_dataset = self.trn_dataloader.dataset
        trn_dataset.load_state_dict(checkpoint['train_dataset_state'])


    def inference(self,
                partial_diffuse_imgs,
                num_inference_steps=50,
                strength=1.0,
                guidance_scale=7.5,
                return_intermediate_images=False):

        intermediate_images = []
        self.eval()
        with torch.no_grad():
            bsz = partial_diffuse_imgs.shape[0]

            # 1) Encode the partial images
            latents = self.model.encode_latents(partial_diffuse_imgs)

            # 2) Scheduler & timesteps
            self.model.noise_scheduler.set_timesteps(num_inference_steps)
            timesteps = self.model.noise_scheduler.timesteps[-int(num_inference_steps * strength):]

            # 3) Add noise
            t_start = timesteps[0]
            noise = torch.randn_like(latents)
            noisy_latents = self.model.noise_scheduler.add_noise(latents, noise, t_start)

            # 4) Prepare embeddings for CFG
            # conditional text + image
            cond_text_embeds = self.prompt_embeds.repeat(bsz, 1, 1).to(latents.device, dtype=self._amp_dtype())
            cond_img_embeds  = self.model.encode_partial(partial_diffuse_imgs).to(latents.device, dtype=self._amp_dtype())

            # unconditional (empty prompt) — same shape
            uncond_text_embeds = self.null_prompt_embeds.repeat(bsz, 1, 1).to(latents.device, dtype=self._amp_dtype())
            uncond_img_embeds  = torch.zeros_like(cond_img_embeds).to(latents.device, dtype=self._amp_dtype())

            # stack them so that batch size doubles
            text_embeds = torch.cat([uncond_text_embeds, cond_text_embeds], dim=0)
            img_embeds  = torch.cat([uncond_img_embeds,  cond_img_embeds],   dim=0)

            # 5) Denoising loop with CFG
            for t in tqdm(timesteps):
                t_tensor = torch.full((2*bsz,), t, device=self.device, dtype=self._amp_dtype())

                # duplicate latents to match doubled batch
                latent_model_input = torch.cat([noisy_latents, noisy_latents], dim=0)

                # model forward once for uncond+cond in one batch
                eps_pair = self.forward(
                    latent_model_input,
                    t_tensor,
                    text_embeds,
                    img_embeds
                )  # should return a tensor of shape [2*bsz, ...]
                
                # split into uncond / cond
                eps_uncond, eps_cond = eps_pair.chunk(2, dim=0)

                # classifier-free guidance
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                # scheduler step on the guided eps
                output = self.model.noise_scheduler.step(eps, t, noisy_latents)
                noisy_latents = output.prev_sample

                if return_intermediate_images:
                    intermediate_images.append(self.model.decode_latents(noisy_latents))

            # final decode
            reconstructed_imgs = self.model.decode_latents(noisy_latents)

        return (reconstructed_imgs, intermediate_images) if return_intermediate_images else reconstructed_imgs

