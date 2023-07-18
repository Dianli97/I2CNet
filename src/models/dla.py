# -*- coding: utf-8 -*-
"""
# @file name  : dla.py
# @author     : Peiji, Chen
# @date       : 2022/09/28
# @brief      :
"""
import torch
import numpy as np
import torch.nn as nn
from typing import List, Union, TypeVar, Tuple, Optional, Callable, Type, Any


# class dla(nn.Module):
#
#     def __init__(self,
#                  in_planes: int,
#                  classes: int,
#                  flag: int = 1,
#                  ) -> None:
#         super(dla, self).__init__()
#         self.in_planes = in_planes
#         self.factor = classes
#         self.flag = flag
#         if self.flag == 1:
#             self.adaptiveAvgPool1d = nn.AdaptiveAvgPool1d(1)
#
#         self.linear1 = nn.Linear(in_features=self.in_planes, out_features=classes)
#         self.softmax1 = nn.Softmax(dim=1)
#
#         self.linear2 = nn.Linear(in_features=classes, out_features=classes)
#         self.softmax2 = nn.Softmax(dim=1)
#
#         self._init_weights()
#
#     def forward(self, x: torch.Tensor, y) -> torch.Tensor:
#
#         if self.flag == 1:
#             out = self.adaptiveAvgPool1d(x)
#             out = torch.flatten(out, 1)
#         else:
#             out = x
#         out = self.linear1(out)
#         out = self.softmax1(out)
#         out = self.linear2(out)
#         out1 = self.softmax2(out)
#         out2 = torch.max(out1, dim=1).values * self.factor
#         # out2 = torch.sum(out1, dim=1)
#         out2 = out2[:, None]
#
#         out2 = y * out2
#         out = torch.add(out1, out2)
#         out = self.softmax1(out)
#
#         return out
#
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight.data, 0, 1)
#                 m.bias.data.zero_()
#
#
# class DLA(nn.Module):
#
#     def __init__(self,
#                  in_planes: int,
#                  classes: int,
#                  block = dla
#                  ):
#         super(DLA, self).__init__()
#
#         self.block1 = block(in_planes=in_planes, classes=classes)
#         self.block2 = block(in_planes=classes, classes=classes, flag=0)
#         self.block3 = block(in_planes=classes, classes=classes, flag=0)
#
#         # self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x, y):
#
#         out = self.block1(x, y)
#         out = self.block2(out, y)
#         out = self.block3(out, y)
#         # out = self.softmax(out)
#
#         return out
class dla(nn.Module):

    def __init__(self, classes):
        super(dla, self).__init__()
        self.factor = classes

        self.linear1 = nn.Linear(in_features=classes, out_features=classes)
        self.softmax1 = nn.Softmax(dim=1)

        self.linear2 = nn.Linear(in_features=classes, out_features=classes)
        self.softmax2 = nn.Softmax(dim=1)

        self._init_weights()

    def forward(self, x: torch.Tensor, y) -> torch.Tensor:

        out = self.linear1(x)
        out = self.softmax1(out)
        out = self.linear2(out)
        out1 = self.softmax2(out)
        out2 = torch.max(out1, dim=1).values * self.factor
        # out2 = torch.sum(out1, dim=1)
        out2 = out2[:, None]

        out2 = y * out2
        out = torch.add(out1, out2)
        out = self.softmax1(out)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()


class DLA(nn.Module):

    def __init__(self,
                 classes: int,
                 block = dla
                 ):
        super(DLA, self).__init__()

        self.block1 = block(classes=classes)
        #self.block2 = block(in_planes=classes, classes=classes, flag=0)
        self.block3 = block(classes=classes)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):

        out = self.block1(x, y)
        #out = self.block2(out, y)
        out = self.block3(out, y)
        # out = self.softmax(out)

        return out

# Dynamic Label Adjustment Strategy
def dynamic_label():
    return DLA(in_planes=33*3, classes=52)


class DLALoss(nn.Module):
    pass


if __name__ == '__main__':

    # step 1: Data
    x = torch.randn(size=(512, 52))
    classes = 52
    batch_size = 512
    label = np.random.randint(0, classes, size=(batch_size, 1))
    true_label = torch.LongTensor(label)
    true_label = torch.zeros(batch_size, classes).scatter_(1, true_label, 1)

    # step 2: Model
    model = DLA(classes=52)

    # step 3: loss function
    print(model(x, true_label))


