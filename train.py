import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()  

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from helpers.dataset import get_dataloaders
from helpers.callbacks import get_callbacks
from helpers.pl_module import GarmentInpainterModule

@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):

    print("-"*50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-"*50)

    pl.seed_everything(cfg.seed)

    logger = WandbLogger(
        project=cfg.logger.project, 
        save_dir="outputs/",
        log_model="all", 
        tags=cfg.logger.tags,
    )

    trn_dataloader, val_dataloader = get_dataloaders(cfg)
    pl_module = GarmentInpainterModule(cfg)
    callbacks = get_callbacks(cfg)

    trainer = pl.Trainer(
        max_steps=cfg.trainer.max_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        gradient_clip_val=cfg.trainer.max_grad_norm,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(pl_module, trn_dataloader, val_dataloader)

if __name__ == "__main__":
    train()