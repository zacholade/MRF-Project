"""
CBAM module code which has been adapted to implement channel reduction.
https://github.com/Jongchan/attention-module
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.util import batched_index_select


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale, scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x, return_scale: bool = False):
        x_out, scale = self.channel_gate(x)

        if return_scale:
            return x_out, scale
        return x_out


class CBAMChannelReduction(nn.Module):
    """
    Uses channel weights to pick top n elements according to reduction parameter.
    Doesn't apply attention to X.
    """
    def __init__(self, seq_len: int, reduction: int):
        super().__init__()
        self.cbam = CBAM(seq_len, reduction_ratio=1)
        self.reduction = reduction

    def forward(self, x):
        batch_size = x.size(0)
        patch_size = x.size(2)

        # Pass through attention module. Calculate scales to pick top 32 channels.
        _, scale = self.cbam(x, return_scale=True)

        # Select the top 32 channels.
        _scale = scale.contiguous().view(batch_size * patch_size * patch_size, -1)
        top_n_indices = torch.topk(_scale, self.reduction, dim=1).indices
        x = x.contiguous().view(batch_size * patch_size * patch_size, -1)
        x = batched_index_select(x, 1, top_n_indices).view(batch_size, self.reduction, patch_size, patch_size)
        return x, scale
