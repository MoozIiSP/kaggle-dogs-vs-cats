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
    shutil.copytree(os.path.join(hydra.utils.get_original_cwd(), 'project'),
                    os.path.join(os.getcwd(), 'code/project'))
    shutil.copy2(os.path.join(hydra.utils.get_original_cwd(), 'hydra_run.py'), os.path.join(os.getcwd(), 'code'))


def collate_fn(batch):
    # FIXME: Performance Error?
    inputs, targets, idxs = tuple(zip(*batch))
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)  # FIXME: .unsqueeze(1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
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

