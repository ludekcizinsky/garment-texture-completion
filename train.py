from omegaconf import OmegaConf
from dotenv import load_dotenv
load_dotenv()  

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from helpers.dataset import get_dataloaders
from helpers.callbacks import get_callbacks

def train():

    print("-"*50)
    cfg = OmegaConf.load("configs/train.yaml")
    print(cfg)
    print("-"*50)

    pl.seed_everything(cfg.seed)

    logger = WandbLogger(
        project=cfg.logger.project, 
        save_dir="outputs/",
        log_model="all", 
        tags=cfg.logger.tags,
    )

    trn_dataloader, val_dataloader = get_dataloaders(cfg)
    diffusion = get_diffusion(cfg)
    model = WeightDiffusionTransformer(cfg)
    pl_module = WeightDenoiser(cfg=cfg, model=model, diffusion=diffusion)
    callbacks = get_callbacks(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(pl_module, trn_dataloader, val_dataloader)

if __name__ == "__main__":
    train()