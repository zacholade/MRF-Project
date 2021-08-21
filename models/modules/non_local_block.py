from typing import Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    """
    Embedded Gaussian Nonlocal block
    """
    def __init__(self, in_channels: int, dimensionality: int,
                 intermediate_dim: int = None, compression: int = 2,
                 bn_layer: bool = True, add_residual: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.dimensionality = dimensionality
        self.compression = compression
        self.bn_layer = bn_layer
        self.add_residual = add_residual

        if not 1 <= self.dimensionality <= 3:
            raise ValueError("Invalid dimensionality.")

        self.inter_dim = min(in_channels // 2, 1) if \
            intermediate_dim is None else intermediate_dim

        if dimensionality == 1:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d(kernel_size=compression)
            bn = nn.BatchNorm1d
        elif dimensionality == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d(kernel_size=(compression, compression))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d(kernel_size=(1, compression, compression))
            bn = nn.BatchNorm3d

        self.w = conv_nd(in_channels=self.inter_dim,
                         out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.w = nn.Sequential(self.w, bn(self.in_channels))

        self.theta = conv_nd(in_channels=in_channels,
                             out_channels=self.inter_dim,
                             kernel_size=1, padding='same', bias=False)
        self.phi = conv_nd(in_channels=in_channels,
                           out_channels=self.inter_dim,
                           kernel_size=1, padding='same', bias=False)

        self.g = conv_nd(in_channels=in_channels,
                         out_channels=self.inter_dim,
                         kernel_size=1, padding='same', bias=False)
        if compression:
            self.g = nn.Sequential(self.g, max_pool)
            self.phi = nn.Sequential(self.g, max_pool)

    def forward(self, x, return_nl_map=False):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_dim, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_dim, *x.size()[2:])
        W_y = self.w(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NonLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels: int, intermediate_dim: int = None,
                 compression: int = 2, bn_layer: bool = True,
                 add_residual: bool = True):
        super().__init__(in_channels, dimensionality=1,
                         intermediate_dim=intermediate_dim,
                         compression=compression,
                         bn_layer=bn_layer, add_residual=add_residual)


class NonLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels: int, intermediate_dim: int = None,
                 compression: int = 2, bn_layer: bool = True,
                 add_residual: bool = True):
        super().__init__(in_channels, dimensionality=2,
                         intermediate_dim=intermediate_dim,
                         compression=compression,
                         bn_layer=bn_layer, add_residual=add_residual)


class NonLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels: int, intermediate_dim: int = None,
                 compression: int = 2, bn_layer: bool = True,
                 add_residual: bool = True):
        super().__init__(in_channels, dimensionality=3,
                         intermediate_dim=intermediate_dim,
                         compression=compression,
                         bn_layer=bn_layer, add_residual=add_residual)


class NonLocalAttention1DFor3D(NonLocalBlock1D):
    """
    Allows for 3D input to have strictly temporal non-local attention applied to it.
    It achieves this by reshaping the spatial dimensions into the batch size,
    passing it through the 1D non-local block before transforming it back to its original shape.
    """
    def forward(self, x):
        batch_size, c, t, h, w, = x.shape
        x = x.view(batch_size * h * w, c, t)
        return super().forward(x).view(batch_size, c, t, h, w)
