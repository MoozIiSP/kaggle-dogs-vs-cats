import importlib
import os
import random
import shutil
from collections import namedtuple
from itertools import product
from typing import Any, Callable, Dict, Generator, List, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)


def set_seed(seed: int = 666):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_useful_info():
    shutil.copytree(os.path.join(hydra.utils.get_original_cwd(), 'src'),
                    os.path.join(os.getcwd(), 'code/src'))
    shutil.copy2(os.path.join(hydra.utils.get_original_cwd(), 'hydra_run.py'), os.path.join(os.getcwd(), 'code'))


def collate_fn(batch):
    # FIXME: Performance Error?
    inputs, targets, idxs = tuple(zip(*batch))
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)  # FIXME: .unsqueeze(1)
    return inputs, targets, idxs


def dump_func(func: Callable):
    """Dump parameters list from function.
    Args:
        func: function."""
    func_struct = {
        'func_name': func.__name__,
        'func_params': dict(),
    }
    # align
    names = func.__code__.co_varnames
    values = func.__defaults__ if func.__defaults__ else tuple()
    offset = len(names) - len(values)
    for i in range(len(names)):
        if i < offset:
            func_struct['func_params'][names[i]] = None
        else:
            func_struct['func_params'][names[i]] = values[i - offset]
    return func_struct


def export_to_submission(data_root: str, preds: Dict):
    from rich.progress import track
    if os.path.exists(os.path.join(data_root, 'sample_submission.csv')):
        df = pd.read_csv(os.path.join(data_root, 'sample_submission.csv'))
    else:
        raise FileNotFoundError
    # FIXME: 😂 Try this, but will remove soon.
    for id, pred in track(preds.items(),
                               description='Write Results'):
        idx = int(os.path.basename(id).split('.')[0]) - 1
        df.loc[idx, 'label'] = pred[1]
    df.to_csv(os.path.join(data_root, 'EvalSubmission.csv'), index=False)


# Load same shape but prefix not same
def reconstruct_state_dict(
    src: Dict, target: Dict) -> Dict:
    r"""Sequentially reconstructing state_dict as same as state_dict of target
    model, so you must be sure that state_dict is suit for your model.
    
    Args:
        state_dict: State dict of model.
        target: State ditc of target model."""
    unmatched_keys = []
    error_msgs = []

    for k1, k2 in zip(*(src.keys(), target.keys())):
        if src[k1].shape == target[k2].shape:
            target[k2] = src[k1]
        else:
            unmatched_keys.append((k1, k2))

    if len(unmatched_keys) > 0:
        error_msgs.insert(
            0, 'Unmacthed key(s) in state_dict:\n{}. '.format(
            '\n'.join(f"{k1} {src[k1].shape} \n\tvs {k2} {target[k2].shape}"
                      for k1, k2 in unmatched_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in reconstructing state_dict: \n\t{}'.format(
                            "\n\t".join(error_msgs)))
    
    class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['unmatched_keys'])):
        def __repr__(self):
            if not self.unmatched_keys:
                return '<All keys reconstructed successfully>'
            return super(_IncompatibleKeys, self).__repr__()

        __str__ = __repr__

    return _IncompatibleKeys(unmatched_keys)


# Refer to https://github.com/PyTorchLightning/pytorch-lightning/pull/1564/files
BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
def _make_trainable(module: torch.nn.Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: torch.nn.Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: torch.nn.Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


# def filter_params(module: torch.nn.Module,
#                   train_bn: bool = True) -> Generator:
#     """Yields the trainable parameters of a given module.
#     Args:
#         module: A given module
#         train_bn: If True, leave the BatchNorm layers in training mode
#     Returns:
#         Generator
#     """
#     children = list(module.children())
#     if not children:
#         if not (isinstance(module, BN_TYPES) and train_bn):
#             for param in module.parameters():
#                 if param.requires_grad:
#                     yield param
#     else:
#         for child in children:
#             for param in filter_params(module=child, train_bn=train_bn):
#                 yield param


# def _unfreeze_and_add_param_group(module: torch.nn.Module,
#                                   optimizer: Optimizer,
#                                   lr: Optional[float] = None,
#                                   train_bn: bool = True):
#     """Unfreezes a module and adds its parameters to an optimizer."""
#     _make_trainable(module)
#     params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
#     optimizer.add_param_group(
#         {'params': filter_params(module=module, train_bn=train_bn),
#          'lr': params_lr / 10.,
#          })

