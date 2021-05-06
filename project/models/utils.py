from collections import OrderedDict, namedtuple
from typing import Any, Dict, List, Optional

import hydra
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
import torch
import torch.nn as nn
from omegaconf import DictConfig
from project.utils import template_utils
from timm.models.factory import create_model
from .head.classifier import ClassifierHead

log = template_utils.get_logger(__name__)


# Load same shape but prefix not same
def reconstruct_state_dict(project: Dict, target: Dict) -> Any:
    r"""Sequentially reconstructing state_dict as same as state_dict of target
    model, so you must be sure that state_dict is suit for your model.

    Args:
        state_dict: State dict of model.
        target: State ditc of target model."""
    unmatched_keys = []
    error_msgs: List[str] = []

    for k1, k2 in zip(*(project.keys(), target.keys())):
        if project[k1].shape == target[k2].shape:
            target[k2] = project[k1]
        else:
            unmatched_keys.append((k1, k2))

    if len(unmatched_keys) > 0:
        error_msgs.insert(
            0,
            "Unmacthed key(s) in state_dict:\n{}. ".format(
                "\n".join(
                    f"{k1} {project[k1].shape} \n\tvs {k2} {target[k2].shape}"
                    for k1, k2 in unmatched_keys
                )
            ),
        )

    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in reconstructing state_dict: \n\t{}".format(
                "\n\t".join(error_msgs)
            )
        )

    class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["unmatched_keys"])):
        def __repr__(self):
            if not self.unmatched_keys:
                return "<All keys reconstructed successfully>"
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


def _recursive_freeze(module: torch.nn.Module, train_bn: bool = True) -> None:
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


def freeze(
    module: torch.nn.Module, n: Optional[int] = None, train_bn: bool = True
) -> None:
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


def rebuild_model(model_cfg: DictConfig, transfer: DictConfig) -> nn.Module:
    """Rebuilding detection head of model for transfer learning."""
    # reconstruct kwargs
    extra_cfg = None
    if model_cfg.extra_cfg:
        log.info(f"using extra configuration to build the model.")
    model = create_model(
        model_cfg.name,
        pretrained=True if transfer.pretrained else False,
        num_classes=model_cfg.num_classes,
        kwargs=extra_cfg,
    )
    if transfer.pretrained:
        log.info(f"Rebuild the {model_cfg.name} model for transfer learning.")
        freeze(model, n=transfer.depth, train_bn=transfer.bn_trainable)
        _layer = list(model.named_children())  # remove fc layer
        end = -1
        for i in range(-1, -5, -1):
            print(i)
            if "pool" in _layer[i][0] or _layer[i][0] == "head":
                end = i
        _layer = _layer[:end]
        backbone = nn.Sequential(OrderedDict(_layer))

        # get last layer output shape
        with torch.no_grad():
            # NOTE: Dummy Input to inference fc shape
            shape = (transfer.in_channels, *transfer.in_size)
            # NOTE: PyTorch 1.8 use SelectAdaptivePool2d which would automately flat the features to fit for linear layer.
            last = int(backbone(torch.randn(size=(1, *shape))).shape[1])

        if transfer.head.layers:
            _layer = []
            for layer in transfer.head.layers:
                if layer.in_features == -1:
                    layer.in_features = last
                _layer.append(hydra.utils.instantiate(layer))
            if transfer.head.last_act:
                _layer.append(hydra.utils.instantiate(transfer.head.last_act))
            head: nn.Module = nn.Sequential(*_layer)
        else:
            # TODO:
            num_channels = []
            if transfer.head.num_channels:
                num_channels.extend(transfer.head.num_channels)
            num_channels.extend([model_cfg.num_classes])
            head = ClassifierHead(in_chs=last, num_channels=num_channels)

        return nn.Sequential(
            OrderedDict(
                [
                    (f"{model_cfg.name}", backbone),
                    ("head", head),
                    ("act", hydra.utils.instantiate(transfer.head.last_act)),
                ]
            )
        )
    else:
        return nn.Sequential(
            OrderedDict(
                [
                    (f"{model_cfg.name}", model),
                    ("act", hydra.utils.instantiate(transfer.head.last_act)),
                ]
            )
        )

