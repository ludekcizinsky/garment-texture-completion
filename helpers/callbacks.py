from tqdm import tqdm
import os
import glob

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.utilities import grad_norm

from torch_ema import ExponentialMovingAverage
import torch
from torch.nn.utils import clip_grad_norm_

def get_callbacks(cfg, exp_name):
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{cfg.checkpoint_dir}/{exp_name}",
        every_n_train_steps=cfg.trainer.checkpoint_every_n_train_steps,
        save_top_k=cfg.trainer.checkpoints_total_limit,
        monitor="step",  # Dummy monitor so it saves by step
        mode="max",
        save_last=True,
    )
    grad_norm_cb = GradNormWithClip(cfg)
    progress_cb = StepProgressBar()
    callbacks = [checkpoint_cb, grad_norm_cb, progress_cb]
    if not cfg.optim.use_cosine_scheduler:
        scheduler_cb = WarmupPlateauScheduler(cfg)
        callbacks.append(scheduler_cb)
    else:
        lr_cb = LearningRateMonitorStep()
        callbacks.append(lr_cb)

    if cfg.optim.use_ema:
        ema_cb = TorchEMACallback(cfg.optim.ema_decay)
        callbacks.append(ema_cb)

    ckpt_path = find_checkpoint_to_resume_from(cfg, exp_name)

    return callbacks, ckpt_path

def find_checkpoint_to_resume_from(cfg, run_name):
    ckpt_path = None
    if cfg.logger.run_id:
        exp_folder = os.path.join(cfg.checkpoint_dir, run_name)
        all_ckpts = glob.glob(os.path.join(exp_folder, "*.ckpt"))
        if all_ckpts:
            ckpt_path = sorted(all_ckpts, key=os.path.getmtime)[-1]

    return ckpt_path


class TorchEMACallback(Callback):
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema = None
        self.ema_state_dict = None

    def on_train_start(self, trainer, pl_module):
        # register model parameters with EMA
        trainable_params = [p for p in pl_module.model.parameters() if p.requires_grad]
        self.ema = ExponentialMovingAverage(trainable_params, decay=self.decay)

        if self.ema_state_dict is not None:
            self.ema.load_state_dict(self.ema_state_dict)

        self.ema.to(pl_module.device, dtype=torch.float32)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # step EMA update
        self.ema.update()

        # Now compute and log the diff each batch
        total_diff = torch.tensor(0.0, device=pl_module.device)
        trainable_params = [p for p in pl_module.model.parameters() if p.requires_grad]
        for param, ema_param in zip(
            trainable_params,
            self.ema.shadow_params
        ):
            ema_p = ema_param.to(param.device)
            total_diff += torch.norm(param.detach() - ema_p)

        # Log per-batch. You can turn prog_bar off if it's too chatty.
        pl_module.log("optim/ema_diff", total_diff, on_step=True, on_epoch=False)


    def on_validation_start(self, trainer, pl_module):
        # copy EMA weights into the model
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

    def on_validation_end(self, trainer, pl_module):
        # restore original parameters
        if self.ema is not None:
            self.ema.restore()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict):
        # Persist the EMA state dict alongside the Lightning checkpoint
        checkpoint["ema_state_dict"] = self.ema.state_dict()
        return checkpoint

    def on_load_checkpoint(self, trainer, pl_module, callback_state: dict):
        self.ema_state_dict = callback_state["ema_state_dict"]


class StepProgressBar(Callback):
    def __init__(self):
        super().__init__()
        self.pbar = None

    def on_train_start(self, trainer, pl_module):
        # total steps you plan to run (from Trainer.max_steps)
        total_steps = trainer.max_steps
        
        # create the bar
        self.pbar = tqdm(total=total_steps, desc="Training", unit="step")

        # if we're resuming, advance to where we left off
        start = trainer.global_step
        if start:
            self.pbar.update(start)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # advance one step each time the optimizer steps
        self.pbar.update(1)

    def on_train_end(self, trainer, pl_module):
        # close it at the very end
        self.pbar.close()


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
        self._resume_state = None


    def on_train_start(self, trainer, pl_module):
        opt = trainer.optimizers[0]

        for pg in opt.param_groups:
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']

        last_step = trainer.global_step - 1
        warmup_last = min(last_step, self.warmup_steps - 1)
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda step: min((step + 1) / self.warmup_steps, 1.0),
            last_epoch=warmup_last,
        )

        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.cfg.optim.plateau_mode,
            patience=self.cfg.optim.plateau_patience,
            factor=self.cfg.optim.plateau_factor,
            min_lr=self.cfg.optim.min_lr,
        )

        if self._resume_state:
            warmup_state = self._resume_state.get("warmup_scheduler")
            plateau_state = self._resume_state.get("plateau_scheduler")

            if warmup_state:
                self.warmup_scheduler.load_state_dict(warmup_state)

            if plateau_state:
                self.plateau_scheduler.load_state_dict(plateau_state)

        print(f"FYI: Warmup scheduler: {self.warmup_scheduler.state_dict()}")
        print(f"FYI: Plateau scheduler: {self.plateau_scheduler.state_dict()}")


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

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save the state of the warmup scheduler
        if hasattr(self, "warmup_scheduler"):
            checkpoint["warmup_scheduler"] = self.warmup_scheduler.state_dict()

        # Save the state of the plateau scheduler
        if hasattr(self, "plateau_scheduler"):
            checkpoint["plateau_scheduler"] = self.plateau_scheduler.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self._resume_state = {
            "warmup_scheduler": checkpoint.get("warmup_scheduler"),
            "plateau_scheduler": checkpoint.get("plateau_scheduler"),
        }

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

class LearningRateMonitorStep(Callback):
    """Logs the learning rate every training step under 'optim/lr'."""

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           outputs, batch, batch_idx: int) -> None:
        # Grab the first optimizer and its first param_group
        optimizer = trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        # Log it (step-wise, no epoch aggregation)
        pl_module.log("optim/lr", lr, on_epoch=False, on_step=True)