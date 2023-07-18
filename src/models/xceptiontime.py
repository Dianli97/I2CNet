# -*- coding: utf-8 -*-
"""
# @file name  : xceptiontime.py
# @author     : Peiji, Chen
# @date       : 2022/08/19
# @brief      :
"""
import torch
from tsai.imports import *
from tsai.models.layers import *
from tsai.models.utils import *


class XceptionModule(Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        ks = [ks // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        self.bottleneck = Conv1d(ni, nf, 1, bias=False) if bottleneck else noop
        self.convs = nn.ModuleList([SeparableConv1d(nf if bottleneck else ni, nf, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return x


@delegates(XceptionModule.__init__)
class XceptionBlock(Module):
    def __init__(self, ni, nf, residual=True, **kwargs):
        self.residual = residual
        self.xception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for i in range(4):
            if self.residual and (i - 1) % 2 == 0: self.shortcut.append(
                BN1d(n_in) if n_in == n_out else ConvBlock(n_in, n_out * 4 * 2, 1, act=None))
            n_out = nf * 2 ** i
            n_in = ni if i == 0 else n_out * 2
            self.xception.append(XceptionModule(n_in, n_out, **kwargs))
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for i in range(4):
            x = self.xception[i](x)
            if self.residual and (i + 1) % 2 == 0: res = x = self.act(self.add(x, self.shortcut[i // 2](res)))
        return x


@delegates(XceptionBlock.__init__)
class XceptionTime(Module):
    def __init__(self, c_in, c_out, nf=16, nb_filters=None, adaptive_size=50, **kwargs):
        nf = ifnone(nf, nb_filters)
        self.block = XceptionBlock(c_in, nf, **kwargs)
        self.head_nf = nf * 32
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size),
                                  ConvBlock(self.head_nf, self.head_nf // 2, 1),
                                  ConvBlock(self.head_nf // 2, self.head_nf // 4, 1),
                                  ConvBlock(self.head_nf // 4, c_out, 1),
                                  GAP1d(1))

    def forward(self, x):
        x = self.block(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    x = torch.randn(2, 10, 20)
    model = XceptionTime(10, 52)
    res = model(x)
    print(res.shape)