import pytorch_lightning as pl
import torch

from helpers.callbacks import prepare_wandb_logger
from helpers.callbacks import prepare_callbacks
from helpers.callbacks import prepare_data_loaders
from helpers.callbacks import prepare_pl_module

from omegaconf import OmegaConf
from dotenv import load_dotenv
load_dotenv()  

def main():

    cfg = OmegaConf.load("configs/train.yaml")

    wandb_logger = prepare_wandb_logger(cfg)
    pl_module = prepare_pl_module(cfg)
    return

    # TODO: resolve seed
    trn_loader, val_loader = prepare_data_loaders(cfg)
    callbacks = prepare_callbacks(cfg)

    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(pl_module, trn_loader, val_loader)

if __name__ == "__main__":
    main()
