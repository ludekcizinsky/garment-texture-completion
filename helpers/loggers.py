import wandb
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf
import os

def get_logger(cfg):

    if not cfg.debug:
        wandb_kwargs = dict(
            project=cfg.logger.project,
            save_dir=cfg.output_dir,
            log_model=False,
            tags=cfg.logger.tags,
        )

        if cfg.logger.run_id:
            wandb_kwargs["id"]     = cfg.logger.run_id
            wandb_kwargs["resume"] = "must"
        logger = WandbLogger(**wandb_kwargs)
        run_name = logger.experiment.name

        # Save the config to the run_id
        path = os.path.join(cfg.output_dir, "configs", logger.experiment.id, "config.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            OmegaConf.save(cfg, f)

    else:
        run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="debug",
            version=run_version,
        )
        run_name = f"debug_{run_version}"

    return logger, run_name