import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()  

import pytorch_lightning as pl

from helpers.loggers import get_logger
from helpers.dataset import get_dataloaders
from helpers.callbacks import get_callbacks
from helpers.pl_module import GarmentInpainterModule

@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def train(cfg: DictConfig):

    if cfg.trainer.max_steps == -1:
        cfg.trainer.max_steps = cfg.max_train_samples // cfg.data.batch_size

    print("-"*50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-"*50)

    pl.seed_everything(cfg.seed)

    logger, run_name = get_logger(cfg)
    trn_dataloader, val_dataloader = get_dataloaders(cfg)
    pl_module = GarmentInpainterModule(cfg)
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