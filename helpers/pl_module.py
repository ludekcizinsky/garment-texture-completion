import os

import torch
import torch.nn as nn
import pytorch_lightning as pl


class GarmentInpainterModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.save_hyperparameters() 

        self.cfg = cfg
        self.model = model

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay,
        )
        
        return {
            "optimizer": optimizer,
        }
