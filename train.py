import hydra
from omegaconf import DictConfig, OmegaConf

from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

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


    if not cfg.debug:
        logger = WandbLogger(
            project=cfg.logger.project,
            save_dir=cfg.output_dir,
            log_model=False,
            tags=cfg.logger.tags,
        )
        exp_name = logger.experiment.name
    else:
        run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="debug",
            version=run_version,
        )
        exp_name = f"debug_{run_version}"

    trn_dataloader, val_dataloader = get_dataloaders(cfg)
    pl_module = GarmentInpainterModule(cfg)
    callbacks = get_callbacks(cfg, exp_name)

    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        max_steps=cfg.trainer.max_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps + 1,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        # avoid epoch based traing
        check_val_every_n_epoch=None, 
        max_epochs=10000,
    )

    trainer.fit(pl_module, trn_dataloader, val_dataloader)

if __name__ == "__main__":
    train()