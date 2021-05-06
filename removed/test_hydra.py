import os
import shutil

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from rich import print
from torchinfo import summary

from project.datamodules.kaggle_dogs_vs_cats_datamodule import DogCatDataModule


@hydra.main(config_path="configs", config_name="config")
def run_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    config = cfg
    callbacks = []
    if "callbacks" in config:
        print("1", config["callbacks"].items())
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                # log.info(f"Instantiating callback <{cb_conf._target_}>")
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    criterions = []
    print(cfg.model.loss)
    for v in cfg.model.loss:
        print(v)
        criterions.append(hydra.utils.instantiate(v))

    print(cfg.datamodule)
    dm = hydra.utils.instantiate(
        cfg.datamodule.datasets,
        transforms = cfg.datamodule.transforms,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        _recursive_=False,
    )
    print("done")
    dm.setup("fit")
    loader = iter(dm.train_dataloader())
    x = next(loader)
    print(x[0].shape, x[1].shape)

    print(cfg.model)
    model = hydra.utils.instantiate(
        cfg.model.net,
        cfg=cfg.model,
        _recursive_=False)
    # summary(model, input_size=[1, 3, 224, 224])

    print(model.configure_optimizers())

if __name__ == "__main__":
    run_model()
