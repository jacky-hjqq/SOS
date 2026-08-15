import os
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from hydra import initialize, compose
from model.pl_seg import Image_Seg_Module

import torch
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    wandb_enable = True
    if wandb_enable:
        # Initial Wandb
        wandb_logger = WandbLogger(
            project="ISeg",        # your project name
            log_model=False        # or True if you want to log model checkpoints
        )
    else:
        wandb_logger = None

    run_id = datetime.now().strftime("%m%d_%H%M%S")
    unique_dirpath = os.path.join("ckpt", run_id)
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        dirpath=unique_dirpath,
        every_n_train_steps=10000,
        save_top_k=-1,  # Save all checkpoints
        monitor=None,
    )

    # Instantiate model
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="train_config")
    model = Image_Seg_Module(config)
    trainer = pl.Trainer(accelerator="gpu",
                        devices=config.devices,
                        max_steps=config.max_steps,
                        precision="16-mixed",
                        accumulate_grad_batches=config.accumulate_grad_batches,
                        callbacks=[checkpoint_callback],
                        strategy=DDPStrategy(find_unused_parameters=True),
                        limit_train_batches=config.limit_train_batches,
                        check_val_every_n_epoch=1,
                        logger=wandb_logger,
                        )
    trainer.fit(model)
