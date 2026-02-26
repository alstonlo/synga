import pathlib
from typing import Literal, Optional

import lightning.pytorch as pl

from src import io


class GradNormMonitor(pl.Callback):

    def on_after_backward(self, trainer, pl_module):
        grad_2norm = pl.utilities.grad_norm(pl_module, norm_type=2.0)[f"grad_2.0_norm_total"]
        pl_module.log("grad_2norm", grad_2norm)


class SimpleTrainer(pl.Trainer):

    def __init__(
        self,
        accelerator: Literal["cpu", "gpu"] = "cpu",
        max_epochs: int = 1000,
        train_steps_per_epoch: Optional[int] = None,
        val_steps_per_epoch: Optional[int] = None,
        check_val_every_n_epoch: int = 1,
        log_every_n_steps: int = 20,
        progress_bar: bool = True,
        wandb: bool = False,
        wandb_dir: pathlib.Path = io.LOG_DIR,
        wandb_project: str = "synga_debug",
        wandb_group: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        checkpoint: bool = False,
        checkpoint_dir: Optional[str] = io.random_checkpoint_dir(),
        early_stop: bool = False,
        early_stop_on: Optional[str] = None,
        early_stop_patience: int = 5,
        verbose: bool = True,
    ):
        callbacks = []

        if checkpoint:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    save_top_k=0,
                    save_last=True,
                    verbose=verbose,
                )
            )
        if early_stop:
            assert early_stop_on is not None
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor=early_stop_on,
                    patience=early_stop_patience,
                    mode=("min" if ("loss" in early_stop_on) else "max"),
                    verbose=verbose,
                )
            )

        if wandb:
            logger = pl.loggers.WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group,
                log_model=False,
                save_dir=wandb_dir,
            )
            callbacks.extend([
                pl.callbacks.LearningRateMonitor(),
                GradNormMonitor(),
            ])
        else:
            logger = False

        super().__init__(
            accelerator=accelerator,
            devices=1,
            callbacks=callbacks,
            enable_checkpointing=checkpoint,
            logger=logger,
            max_epochs=max_epochs,
            limit_train_batches=train_steps_per_epoch,
            limit_val_batches=val_steps_per_epoch,
            check_val_every_n_epoch=check_val_every_n_epoch,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=(verbose and progress_bar),
            enable_model_summary=verbose,
        )
