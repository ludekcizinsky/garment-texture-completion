import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache/"
os.environ["WANDB_CACHE_DIR"] = "/scratch/izar/cizinsky/.cache/wandb"


import hydra
from omegaconf import DictConfig

from dotenv import load_dotenv
load_dotenv()  

import pytorch_lightning as pl

from helpers.loggers import get_logger
from helpers.dataset import get_dataloaders
from helpers.callbacks import get_callbacks
from helpers.pl_module import GarmentInpainterModule
from helpers.config import get_correct_config

@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):

    cfg = get_correct_config(cfg)
    pl.seed_everything(cfg.seed)

    logger, run_name = get_logger(cfg)
    trn_dataloader, val_dataloader = get_dataloaders(cfg)
    pl_module = GarmentInpainterModule(cfg, trn_dataloader)
    callbacks, ckpt_path = get_callbacks(cfg, run_name)

    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        max_steps=cfg.trainer.max_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=False,
        # avoid epoch based traing
        check_val_every_n_epoch=None, 
        max_epochs=10000,
    )

    trainer.fit(pl_module, trn_dataloader, val_dataloader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    train()