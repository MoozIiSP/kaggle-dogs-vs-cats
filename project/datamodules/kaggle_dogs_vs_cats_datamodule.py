import logging
from os.path import join as opj

from torch.utils.data.dataset import Subset
from project.datamodules.datasets.utils import testset_collate_fn
from typing import Dict, Optional, Tuple
from collections import Counter

import hydra
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms as T

from .datasets.generic import (GenericClassifierDataset,
                               GenericTestEvaluationDatset)
from .datasets.utils import testset_collate_fn

logger = logging.getLogger(__name__)


class DogCatDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "datasets/",
        n_channels: int = 3,
        crop_size: int = 224,
        transforms: Optional[Dict] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        self.data_dir = data_dir
        # self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # from configs

        assert (
            transforms.type == "torchvision"
        ), "Augmentation method must be torchvision."
        self.transforms = transforms

        # self.dims is returned when you call datamodule.size()
        self.dims = (n_channels, *crop_size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        logger.info("DogCatDataModule Initialized.")

    def setup(self, stage: Optional[str] = None):
        """
        NOTE: dont apply transform onto validset and testset.
        """
        if stage == "fit":
            dataset = GenericClassifierDataset(data_dir=opj(self.data_dir, "train"))
            n_images = len(dataset)

            # NOTE: construct transform for trainset and validset
            trainset_aug = []
            for v in self.transforms.train:
                trainset_aug.append(hydra.utils.instantiate(v))
            eval_aug = []
            for v in self.transforms.eval:
                eval_aug.append(hydra.utils.instantiate(v))

            self.data_train, self.data_val = random_split(
                dataset, [int(n_images * 0.8), n_images - int(n_images * 0.8)]
            )
            # NOTE: Set Augmentation method for every subset
            self.data_train.dataset.transform = T.Compose(trainset_aug)
            self.data_val.dataset.transform = T.Compose(eval_aug)
        elif stage == "test":
            eval_aug = []
            for v in self.transforms.eval:
                eval_aug.append(hydra.utils.instantiate(v))
            self.data_test = GenericTestEvaluationDatset(
                data_dir=opj(self.data_dir, "test"), transform=T.Compose(eval_aug)
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=testset_collate_fn
        )
