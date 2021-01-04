import os
import sys
from typing import Dict, Tuple

sys.path.append('/home/mooziisp/GitRepos/DLToolbox/application/Repos/Kaggle-dogs-vs-cats')

import numpy as np
import torch
import albumentations as A
from albumentations.core.composition import Compose
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split
from src.utils.utils import load_obj
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class KaggleDataset(Dataset):

    def __init__(self,
                 cfg: DictConfig = None,
                 data_ids: list = None,
                 mode: str = 'trainval',
                 transforms: Compose = None):
        """
        Prepare data for Kaggle Competition.
        """
        self.cfg = cfg
        self.data_ids = data_ids
        self.categories = self.cfg.data.categories
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, idx: int):
        data_ind = self.data_ids[idx]

        imp = os.path.join(self.cfg.data.root_path,
                           self.mode, data_ind)
        image = np.array(Image.open(imp), dtype=np.float32)
        if self.mode != 'test':
            label = torch.tensor(self.categories[data_ind.split('.')[0]], dtype=torch.long)

        # normalization.
        if np.max(image) > 1:
            image /= 255.0

        if self.mode != 'test':
            # for train and valid test create target dict.
            data_dict = {
                'image': image,
            }
            image = self.transforms(**data_dict)['image']
        else:
            data_dict = {
                'image': image,
            }
            image = self.transforms(**data_dict)['image']
       
        if self.mode != 'test':
            return image, label, data_ind
        else:
            return image, data_ind

    def __len__(self) -> int:
        return len(self.data_ids)


def unzip(fr, to):
    from zipfile import ZipFile
    if os.path.isdir(to):
        pass
    else:
        os.mkdir(to)
    with ZipFile(fr) as f:
        for name in f.namelist():
            f.extract(name, to)


def get_trainval_dataset(cfg: DictConfig) -> dict:
    def is_valid_file(fp: str):
        def is_valid_size(size: Tuple, threshold: Tuple = (16, 16)):
            return size[0] > threshold[0] and size[1] > threshold[1]

        im = Image.open(os.path.join(cfg.data.root_path, 'trainval', fp))
        return is_valid_size(im.size)

    root_dir = f'{cfg.data.root_path}'
    # check existed cache index file
    if os.path.exists(os.path.join(root_dir, 'trainval.txt')):
        with open(os.path.join(root_dir, 'trainval.txt'), 'r') as f:
            data_ids = [line.strip() for line in f.readlines()]
    else:
        data_ids = [
            fname for fname in os.listdir(f'{root_dir}/trainval')
        ]
        # save index to cache file
        with open(os.path.join(root_dir, 'trainval.txt'), 'w') as f:
            f.writelines('\n'.join(filter(is_valid_file, data_ids)))
    assert len(data_ids) > 0, \
        f"found {len(data_ids)} data. please remove cache file first."

    train_ids, valid_ids = train_test_split(
        data_ids, test_size=0.2, random_state=cfg.training.seed)

    # for fast training
    if cfg.training.debug:
        train_ids = train_ids[:10]
        valid_ids = valid_ids[:10]

    # dataset
    dataset_class = load_obj(cfg.dataset.class_name)

    # initialize augmentations
    train_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['train']['augs']]
    train_augs = A.Compose(train_augs_list)

    valid_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['valid']['augs']]
    valid_augs = A.Compose(valid_augs_list)

    train_dataset = dataset_class(cfg,
                                  train_ids,
                                  'trainval',
                                  train_augs)

    valid_dataset = dataset_class(cfg,
                                  valid_ids,
                                  'trainval',
                                  valid_augs)

    return {'train': train_dataset, 'valid': valid_dataset}


def get_test_dataset(cfg: DictConfig):
    def is_valid_file(fp: str):
        def is_valid_size(size: Tuple, threshold: Tuple = (16, 16)):
            return size[0] > threshold[0] and size[1] > threshold[1]

        im = Image.open(os.path.join(cfg.data.root_path, 'test', fp))
        return is_valid_size(im.size)

    root_dir = f'{cfg.data.root_path}'
    # check existed cache index file
    if os.path.exists(os.path.join(root_dir, 'test.txt')):
        with open(os.path.join(root_dir, 'test.txt'), 'r') as f:
            data_ids = [line.strip() for line in f.readlines()]
    else:
        data_ids = [
            fname for fname in os.listdir(f'{root_dir}/test')
        ]
        # save index to cache file
        with open(os.path.join(root_dir, 'test.txt'), 'w') as f:
            f.writelines('\n'.join(filter(is_valid_file, data_ids)))
    assert len(data_ids) > 0, \
        f"found {len(data_ids)} data. please remove cache file first."

    data_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['test']['augs']]
    data_augs = A.Compose(data_augs_list)
    dataset_class = load_obj(cfg.dataset.class_name)

    test_dataset = dataset_class(cfg,
                                 data_ids,
                                 'test',
                                 data_augs)
    return test_dataset


def plot_tensors(tensors: torch.Tensor, targets: Dict, figsize: Tuple[int, int], row: int = 4):
    def to_cpu(t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu()
    bs = tensors.size(0)
    plt.figure(figsize=figsize)
    fig, axes = plt.subplots(bs//row, row, sharey=False)
    for i in range(1, bs):
        axes[i//row, i%row].imshow(to_cpu(tensors.index_select(0, i)).permute(1, 2, 0))
        for box in targets['boxes']:
            orig_xy = (box[0], box[1])
            w, h = box[2] - box[0], box[3] - box[1]
            rect = Rectangle(orig_xy, w, h, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            axes[i//row, i%row].add_patch(rect)
    plt.show()


if __name__ == "__main__":
    import hydra
    import matplotlib.pyplot as plt
    import torch
    import tqdm
    
    @hydra.main(config_path='/home/mooziisp/GitRepos/DLToolbox/application/Repos/Kaggle-dogs-vs-cats/conf', 
                config_name='config.yaml')
    def m(cfg: DictConfig):
        # TODO: remove cache index file
        datasets = get_trainval_dataset(cfg)
        trainset = datasets['train']
        dataloader = torch.utils.data.DataLoader(trainset, batch_size=16, num_workers=0)
        for i, (x, y, ind) in tqdm.tqdm(enumerate(dataloader)):
            print(f'{ind}: ')
            print(f'x: {x.shape} {x.dtype}')
            print(f'y: {y.shape} {y.dtype}')
            break
        print('test trainloader done.')
        validset = datasets['valid']
        dataloader = torch.utils.data.DataLoader(validset, batch_size=16, num_workers=0)
        for i, (x, y, ind) in tqdm.tqdm(enumerate(dataloader)):
            print(f'{ind}: ')
            print(f'x: {x.shape} {x.dtype}')
            print(f'y: {y.shape} {y.dtype}')
            break
        print('test trainloader done.')
        testset = get_test_dataset(cfg)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=16, num_workers=0)
        for i, (x, ind) in tqdm.tqdm(enumerate(dataloader)):
            print(f'{ind}[{len(testset)}]: ')
            print(f'x: {x.shape} {x.dtype}')
            break
        print('test testloader done.')
    m() # antlr4, importlib_resources
