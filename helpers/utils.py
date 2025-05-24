import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def get_optimizer(cfg, model, return_dict):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.optim.weight_decay
    )

    return_dict["optimizer"] = optimizer

    return return_dict

def get_lr_scheduler(cfg, return_dict):

    optimizer = return_dict["optimizer"]

    if cfg.optim.use_cosine_scheduler:
        # Make your the two schedulers
        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cfg.trainer.max_steps - cfg.optim.warmup_steps,
            eta_min=cfg.optim.min_lr
        )

        # Chain them *sequentially*, switching at warmup_steps
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.optim.warmup_steps]
        )

        return_dict["lr_scheduler"] = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

    return return_dict