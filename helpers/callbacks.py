from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def get_callbacks(cfg, exp_name):
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"{cfg.checkpoint_dir}/{exp_name}",
        every_n_train_steps=cfg.trainer.checkpoint_every_n_train_steps,
        save_top_k=cfg.trainer.checkpoints_total_limit,
        monitor="step",  # Dummy monitor so it saves by step
        mode="max",
    )


    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_cb, lr_monitor]

    return callbacks
