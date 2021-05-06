import os
import warnings
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class GenericClassifierDataset(ImageFolder):
    """Example dataset class for loading images from folder."""

    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        super(GenericClassifierDataset, self).__init__(
            root=data_dir,
            transform=transform,
            is_valid_file = None,  # filter invalid file
        )
        self.imgs = self.samples

class GenericTestEvaluationDatset(Dataset):

    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        super(GenericTestEvaluationDatset, self).__init__()
        if transform:
            warnings.warn("You shouldn't apply augmentation for testset.")
            self.transform = transform
        self.images = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, (idx, self.images[idx])

    def __len__(self):
        return len(self.images)


class GenericSegmentationDataset(Dataset):
    """Generic Segmentation Dataset"""

    def __init__(self, dir: str, transform: Optional[Callable] = None):
        self.transform = transform
        self.images = [os.path.join(dir, fname) for fname in os.listdir(dir)]

    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert("L")  # convert to black and white
        # image = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)