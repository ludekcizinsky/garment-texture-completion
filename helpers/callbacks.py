from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import grad_norm

import torch
from torch.nn.utils import clip_grad_norm_

def get_callbacks(cfg, exp_name):
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{cfg.checkpoint_dir}/{exp_name}",
        every_n_train_steps=cfg.trainer.checkpoint_every_n_train_steps,
        save_top_k=cfg.trainer.checkpoints_total_limit,
        monitor="step",  # Dummy monitor so it saves by step
        mode="max",
    )
    grad_norm_cb = GradNormWithClip(cfg)
    scheduler_cb =  WarmupPlateauScheduler(cfg)
    callbacks = [checkpoint_cb, scheduler_cb, grad_norm_cb]

    return callbacks


class WarmupPlateauScheduler(Callback):
    def __init__(self, cfg):
        """
        Args:
            warmup_epochs: number of epochs to linearly increase LR from 0 → base_lr
            plateau_kwargs: kwargs for torch.optim.lr_scheduler.ReduceLROnPlateau
                            (e.g. {'mode':'min','patience':2,'factor':0.1,'min_lr':1e-6})
        """
        super().__init__()
        self.cfg = cfg
        self.warmup_steps = self.cfg.optim.warmup_steps

    def on_train_start(self, trainer, pl_module):
        opt = trainer.optimizers[0]

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda step: min((step + 1) / self.warmup_steps, 1.0)
        )

        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.cfg.optim.plateau_mode,
            patience=self.cfg.optim.plateau_patience,
            factor=self.cfg.optim.plateau_factor,
            min_lr=self.cfg.optim.min_lr,
        )


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step < self.warmup_steps:
            self.warmup_scheduler.step()

        current_lr = trainer.optimizers[0].param_groups[0]["lr"]
        pl_module.log("optim/lr", current_lr, on_step=True)


    def on_validation_epoch_end(self, trainer, pl_module):
        global_step = trainer.global_step
        if global_step >= self.warmup_steps:
            metric = self.cfg.optim.plateau_metric
            val_metric = trainer.callback_metrics.get(metric)
            if val_metric is not None:
                self.plateau_scheduler.step(val_metric)


class GradNormWithClip(Callback):
    def __init__(self, cfg):
        """
        Args:
            max_norm: the clipping threshold (same semantics as `gradient_clip_val`)
            norm_type: p-norm degree
        """
        self.max_norm = cfg.optim.max_grad_norm
        self.norm_type = cfg.optim.grad_norm_type

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # 1) Pre-clip norm from Lightning util
        norms = grad_norm(pl_module, norm_type=self.norm_type)
        pre = norms[f"grad_{self.norm_type}_norm_total"]

        # 2) Do the clip ourselves (in-place on p.grad)
        clip_grad_norm_(pl_module.parameters(), self.max_norm, self.norm_type)

        # 3) Compute post-clip norm from the same util
        norms_after = grad_norm(pl_module, norm_type=self.norm_type)
        post = norms_after[f"grad_{self.norm_type}_norm_total"]

        # 4) Log both
        pl_module.log("optim/grad_norm_preclip", pre, on_epoch=False, on_step=True)
        pl_module.log("optim/grad_norm_postclip", post, on_epoch=False, on_step=True)