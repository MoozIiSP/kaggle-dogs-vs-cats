import sys
from collections import OrderedDict
from typing import Dict, List, Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from project.utils import template_utils
from project.metrics.kaggle import KagglePredRecord, LogLoss
from torchmetrics.collections import MetricCollection
from torchmetrics import Accuracy

from .utils import rebuild_model

log = template_utils.get_logger(__name__)


# NOTE: recommended one task one module
# TODO: rename to KaggleDogCatLitModule
class ClassifierLitModel(pl.LightningModule):
    """Generic Classifier model"""

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super(ClassifierLitModel, self).__init__()
        self.save_hyperparameters()
        self.model = rebuild_model(cfg.net, cfg.transfer)
        # criterion
        self.criterions = []
        for v in cfg.loss:
            self.criterions.append(hydra.utils.instantiate(v))
        # Metrics
        self.metrics = MetricCollection(
            {
                "val/acc": Accuracy(),
                "val/logloss": LogLoss(),
            }
        )
        self.kaggle_metrics = {
            "kaggle": KagglePredRecord,
        }
        self.metric_hist: Dict[str, List] = {
            "train/loss": [],
            "val/acc": [],
            "val/loss": [],
            "val/logloss": [],
        }

        log.info("The lightning model initialized.")

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optims = []
        scheds = []
        for group in self.hparams.cfg.optimizer.groups:
            (_, optim), (_, sched) = group.items()
            optim = hydra.utils.instantiate(optim, params=self.parameters())
            optims.append(optim)
            if sched:
                scheds.append(hydra.utils.instantiate(sched, optimizer=optim))
        if scheds:
            return optims, scheds  # FIXME: scheds only work some schedulers.
        return optim

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        losses = 0
        for criterion in self.criterions:
            losses += criterion(y_hat, y)

        self.log("train/loss", losses, on_step=True)

        return losses

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        losses = 0
        for criterion in self.criterions:
            losses += criterion(y_hat, y)

        self.log("val/loss", losses)
        self.log(
            "val/acc", self.metrics["val/acc"](y_hat, y), on_epoch=True, prog_bar=True
        )
        self.log(
            "val/logloss",
            self.metrics["val/logloss"](
                y_hat, F.one_hot(y, num_classes=self.hparams.cfg.net.num_classes)
            ),
        )

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.metric_hist["val/logloss"].append(
            self.trainer.callback_metrics["val/logloss"]
        )
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)
        self.log(
            "val/logloss_best", min(self.metric_hist["val/logloss"]), prog_bar=False
        )

    def test_step(self, batch, batch_idx) -> None:  # type: ignore[override]
        x, meta = batch
        preds = self(x).detach().cpu().numpy()
        # FIXME:
        self.kaggle_metrics["kaggle"].update(meta, preds)
