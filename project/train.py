from typing import List, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from pytorch_lightning.loggers import LightningLoggerBase
from torchinfo import summary

from project.utils import template_utils

log = template_utils.get_logger(__name__)


def train(cfg: DictConfig):
    if "seed" in cfg:
        pl.seed_everything(cfg.seed)

    # Instantiate all modules specified in the configs
    log.info(f"Instantiating datamodule <{cfg.datamodule.datasets._target_}>")
    datamodule: LightningDataModule = \
        hydra.utils.instantiate(
            cfg.datamodule.datasets,
            transforms=cfg.datamodule.transforms,
            _recursive_=False,
        )

    log.info(f"Instantiating model <{cfg.model.net._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model.net,
        cfg=cfg.model,
        _recursive_=False,
    )

    log.info("the architecture: \n" + model.__str__())
    input_size = [1, cfg.datamodule.datasets.n_channels, cfg.datamodule.datasets.crop_size, cfg.datamodule.datasets.crop_size]
    log.info(f"feed {input_size} into the model summary: \n")
    summary(model, input_size=input_size)

    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger: List[LightningLoggerBase] = []
    if "logger" in cfg:
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    template_utils.log_hyperparameters(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # # Evaluate model on test set after training
    # if not cfg.trainer.get("fast_dev_run"):
    #     log.info("Starting testing!")
    #     trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    template_utils.finish(
        config=cfg,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for Optuna optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
