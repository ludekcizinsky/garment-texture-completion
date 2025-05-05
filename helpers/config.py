import os
from omegaconf import OmegaConf

def get_correct_config(cfg):
    # Load the config from the run_id
    if cfg.logger.run_id:
        path = os.path.join(cfg.output_dir, "configs", cfg.logger.run_id, "config.yaml")
        run_id = cfg.logger.run_id
        max_train_samples = cfg.max_train_samples
        with open(path, "r") as f:
            cfg = OmegaConf.load(f)
        cfg.logger.run_id = run_id
        cfg.max_train_samples = max_train_samples
        cfg.trainer.max_steps = -1

    # Compute dynamically the max_steps as a function of the number of samples and the batch size
    if cfg.trainer.max_steps == -1:
        cfg.trainer.max_steps = cfg.max_train_samples // cfg.data.batch_size

    # Print the config
    print("-"*50)
    print(OmegaConf.to_yaml(cfg))  # print config to verify
    print("-"*50)

    return cfg