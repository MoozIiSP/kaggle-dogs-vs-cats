from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Take care of Unbalanced Dataset very well"""
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, smooth=1., reduction='mean', ignore_index=[]):
        try:
            assert y_true.dtype == torch.int64, f"y_true: {y_true.dtype} vs torch.int64"
        except AssertionError:
            y_true = torch.as_tensor(y_true, dtype=torch.int64)
        assert isinstance(ignore_index, list), f"ignore_index must be list"
        bs = y_pred.size(0)
        nc = y_pred.size(1)

        # y_pred = torch.sigmoid(y_pred)  # project into [0, 1]
        if y_true.ndim == 3:
            y_true = F.one_hot(y_true[:, ...], num_classes=nc).permute(0, 3, 1, 2)  # one_hot for ground-truth

        index = list(set(list(range(nc))) - set(ignore_index))
        y_pred = y_pred[:, index].contiguous().view(bs, -1)
        y_true = y_true[:, index].contiguous().view(bs, -1)
        dcs = ((2. * y_pred * y_true).sum() + smooth) / (
                (y_pred ** 2).sum() + (y_true ** 2).sum() + smooth)

        if reduction == 'none':
            return 1. - dcs
        elif reduction == 'mean':
            return (1. - dcs).mean()
        elif reduction == 'sum':
            return (1. - dcs).sum()
        else:
            raise RuntimeError