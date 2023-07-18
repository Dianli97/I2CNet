import torch
import numpy as np
import torch.nn as nn
from typing import List, Union, TypeVar, Tuple, Optional, Callable, Type, Any


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
        self.block2 = block(classes=classes)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):

        out = self.block1(x, y)
        out = self.block2(out, y)
        return out

# Dynamic Label Adjustment Strategy
def dynamic_label():
    return DLA(in_planes=33*3, classes=52)
