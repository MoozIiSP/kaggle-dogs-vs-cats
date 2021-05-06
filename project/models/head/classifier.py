import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from timm.models.layers.classifier import _create_fc, _create_pool


class ClassifierHead(torch.nn.Module):
    """timm model modified version"""

    def __init__(
        self,
        in_chs,
        num_channels: List = None,
        pool_type="avg",
        drop_rate=0.0,
        use_conv=False,
    ):
        super(ClassifierHead, self).__init__()
        num_classes = num_channels[-1]

        self.drop_rate = drop_rate
        self.global_pool, num_pooled_features = _create_pool(
            in_chs, num_classes, pool_type, use_conv=use_conv
        )

        mid_chs = num_pooled_features
        self.fcs = nn.ModuleList()
        for chs in num_channels:
            self.fcs.append(_create_fc(mid_chs, chs, use_conv=use_conv))
            mid_chs = chs

        self.flatten_after_fc = use_conv and pool_type

    def forward(self, x):
        x = self.global_pool(x)
        for fc in self.fcs:
            if self.drop_rate:
                x = F.dropout(x, p=float(self.drop_rate), training=self.training)
            x = fc(x)
        return x
