import sys
from collections import OrderedDict


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from project.utils.dataset import get_test_dataset, get_trainval_dataset
from project.utils.metric import KagglePredRecord, LogLoss
from project.utils.utils import collate_fn, freeze, load_obj
from torchvision import models


class LitTransferLearning(pl.LightningModule):

    def __init__(self, cfg: DictConfig) -> None:
        super(LitTransferLearning, self).__init__()
        self.hparams = cfg
        self.model = self._build_model()
        # Hyper-parameters
        self.batch_size = self.hparams.data.batch_size
        # Loss function
        self.criterion = load_obj(self.hparams.loss.class_name)
        # Metrics
        self.train_logloss = LogLoss()
        self.valid_logloss = LogLoss()
        self.accuracy = pl.metrics.Accuracy()
        self.submission = KagglePredRecord()

    def _build_model(self) -> nn.Sequential:
        model_fn = getattr(models, self.hparams.model.class_name)
        if self.hparams.model.params.pretrained:
            backbone = model_fn(pretrained=self.hparams.model.params.pretrained)

            _layer = list(backbone.named_children())[:self.hparams.model.depth]
            backbone = nn.Sequential(OrderedDict(_layer))
            freeze(backbone, train_bn=self.hparams.model.bn_trainable)

            # NOTE: 0 is stand for flatten tensor, otherwise don't need flatten.
            if self.hparams['model']['head']['last'] == 0:
                # get last layer output shape
                with torch.no_grad():
                    # NOTE: Dummy Input to inference fc shape
                    shape = backbone(torch.randn(1, 3, 224, 224)).shape[1:]
                    self.hparams['model']['head']['last'] = shape[0] * shape[1] * shape[2]
            head = nn.Sequential(
                *[load_obj(i['class_name'])(**i['params']) for i in self.hparams['model']['head']['layers']]
            )

            return nn.Sequential(OrderedDict([
                (f'{self.hparams.model.class_name}_backbone', backbone),
                ('head', head)
            ]))
        else:
            raise KeyError("Please set pretrained to True.")

    def forward(self, x) -> torch.Tensor:
        out = self.model[0](x)
        if isinstance(self.model[1][0], nn.Linear):
            out = self.model[1](torch.flatten(out, 1))
        else:  # FIXME: dirty fix for squeezenet and other classifier layer is not consist of nn.Linear.
            out = torch.flatten(self.model[1](out), 1)
        return out  # NOTE Don't use both softmax and sigmoid

    def prepare_data(self) -> None:
        trainval = get_trainval_dataset(self.hparams)
        self.trainset = trainval['train']
        self.validset = trainval['valid']
        self.testset = get_test_dataset(self.hparams)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.trainset,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.hparams.data.num_workers,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(self.validset,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.hparams.data.num_workers,
                                                   shuffle=False,
                                                   collate_fn=collate_fn)
        return valid_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.testset,
                                                  batch_size=self.batch_size,
                                                  num_workers=self.hparams.data.num_workers,
                                                  shuffle=False)
        return test_loader

    def configure_optimizers(self):
        optimizer = load_obj(self.hparams.optimizer.class_name)(self.model.parameters(),
                                                                **self.hparams.optimizer.params)
        scheduler = load_obj(self.hparams.scheduler.class_name)(optimizer, **self.hparams.scheduler.params)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        losses = self.criterion(y_hat, y.type_as(y_hat))

        # self.log('train_loss', losses, on_step=True, prog_bar=False, logger=True)
        # self.log('train_log_loss', self.train_logloss(y_hat, F.one_hot(y)),
        #          prog_bar=False, logger=True)

        return losses

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)  # BUG bceloss dont support automixer
        losses = self.criterion(y_hat, y.type_as(y_hat))

        # self.log('valid_loss', losses, on_epoch=True, prog_bar=False, logger=True)
        self.log('valid_acc', self.accuracy(y_hat, y), on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True)
        # self.log('valid_log_loss', self.valid_logloss(y_hat, F.one_hot(y)),
        #          on_epoch=True, prog_bar=True, logger=True)

        return losses

    def validation_epoch_end(self, outputs) -> None:
        self.log('avg_loss', torch.stack([x for x in outputs]).mean(),
                 on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx) -> None:
        x, im_ids = batch
        preds = self(x).detach().cpu().numpy()
        self.submission.update(im_ids, preds)
