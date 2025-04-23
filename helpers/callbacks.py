from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from helpers.pl_module import GarmentInpainterModule

def prepare_wandb_logger(cfg):

    # todo: possibly add description and tags
    wandb_logger = WandbLogger(
        entity="ludekcizinsky",
        project="pbr-generation", 
        log_model="checkpoints",  # important for uploading best model
        tags=cfg.tags, 
    )

    return wandb_logger


def prepare_callbacks(cfg):

    checkpoint_cl = ModelCheckpoint(
        monitor=cfg.monitor_metric,       # e.g., "val_loss" or "val_acc"
        mode=cfg.monitor_mode,            # "min" or "max"
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.4f}",
        auto_insert_metric_name=False
    )

    #TODO: feel fre to look into more callbacks, e.g. EarlyStopping or LearningRateMonitor

    callbacks = [checkpoint_cl]

    return callbacks

def prepare_data_loaders(cfg):

    #TODO: implement this using their dataloading 
    trn_dataloader = ...
    val_dataloader = ...


def prepare_pl_module(cfg):
    pl_module = GarmentInpainterModule(cfg)
    return pl_module