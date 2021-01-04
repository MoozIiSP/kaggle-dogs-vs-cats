import os
import shutil

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from src.lightning_classes.lightning_transfer import LitTransferLearning
from src.utils.utils import export_to_submission, save_useful_info, set_seed


def run(cfg: DictConfig):
    """
    Run pytorch-lightning model

    Args:
        cfg: hydra config

    Returns:

    """
    set_seed(cfg.training.seed)

    model = LitTransferLearning(hparams=cfg)
    if cfg.training.resume:
        ckpt = cfg.training.checkpoint
        model.load_from_checkpoint(ckpt)

    early_stopping = pl.callbacks.EarlyStopping(**cfg.callbacks.early_stopping.params)
    lr_monitor = pl.callbacks.LearningRateMonitor(**cfg.callbacks.lr_monitor.params)
    gpu_monitor = pl.callbacks.GPUStatsMonitor(**cfg.callbacks.gpu_monitor.params)
    # model_checkpoint = pl.callbacks.ModelCheckpoint(**cfg.callbacks.model_checkpoint.params)

    logger = [TensorBoardLogger(save_dir=cfg.general.save_dir)]
    # FIXME: Bad Network for Comet-ML
    # if not cfg.trainer.fast_dev_run:
    #     logger.append(
    #         CometLogger(save_dir=cfg.general.save_dir,
    #                     workspace=cfg.general.workspace,
    #                     project_name=cfg.general.project_name,
    #                     api_key=cfg.private.comet_api,
    #                     experiment_name=os.getcwd().split('\\')[-1])
    #     )

    trainer = pl.Trainer(logger=logger,
                         callbacks=[early_stopping, lr_monitor, gpu_monitor],
                         # nb_sanity_val_steps=0,
                         # gradient_clip_val=0.5,
                         **cfg.trainer)
    # BUG: trainer.tune(model)

    trainer.fit(model)

    # test
    trainer.test(model)
    export_to_submission(
        cfg.data.root_path,
        model.submission.preds)

    # save as a simple torch model
    backslash = '\\'
    model_name = f"{os.getcwd().split(backslash)[-1]}.pth"
    torch.save(model.model.state_dict(), model_name)


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    save_useful_info()
    run(cfg)


if __name__ == "__main__":
    run_model()
