
import torch
import torch.nn as nn
from typing import List, Union, TypeVar, Tuple, Optional, Callable, Type, Any

T = TypeVar('T')


def convnxn(in_planes: int, out_planes: int, kernel_size: Union[T, Tuple[T]], stride: int = 1,
            groups: int = 1, dilation=1) -> nn.Conv1d:
    """nxn convolution and input size equals output size
    O = (I-K+2*P) / S + 1
    """
    if stride == 1:
        k = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding_size = int((k - 1) / 2)  # s = 1, to meet output size equals input size
        return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_size, dilation=dilation,
                         groups=groups, bias=False)
    elif stride == 2:
        k = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding_size = int((k - 2) / 2)  # s = 2, to meet output size equals input size // 2
        return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_size, dilation=dilation,
                         groups=groups, bias=False)
    else:
        raise Exception('No such stride, please select only 1 or 2 for stride value.')


# ===================== DWResBlock ==========================
# The DepthWise Residual Block is used to ensure we can train a
# deep network and also make the data flow more efficient
# input  : (eg. [batch_size, 10, 20])   groups=10, group_width=12, in_planes=36.
# output : [batch_size, 48, 500]         in_planes=48
# the input size and output size should be an integer multiple of groups
# ===================== DWResBlock ==========================

class DWResBlock(nn.Module):
    expansion: int = 3

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: Union[T, Tuple[T]],
            stride: int = 1,
            groups: int = 1,  # 10 sparse sEMG channel
            input_length: int = 20, # window length
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DWResBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.groups = groups
        self.input_length = input_length
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1x1_1 = convnxn(in_planes, out_planes, kernel_size=1, stride=1, groups=groups)
        self.bn1 = norm_layer(out_planes)
        self.conv3x3 = convnxn(out_planes, out_planes, kernel_size=3, stride=stride, groups=groups)
        self.bn2 = norm_layer(out_planes)
        self.conv1x1_2 = convnxn(out_planes, out_planes * self.expansion, kernel_size=1, stride=1, groups=groups)
        self.bn3 = norm_layer(out_planes * self.expansion)
        self.act = nn.SELU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.downsample = nn.Sequential(
                convnxn(in_planes, out_planes * self.expansion, kernel_size=1, stride=stride, groups=groups),
                norm_layer(out_planes * self.expansion)
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.conv1x1_1(x)
        out = self.bn1(out)

        out = self.conv3x3(out)
        out = self.bn2(out)

        out = self.dropout(out)

        out = self.conv1x1_2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


# ===================== ChannelInterBlock1st ===========================
# The Channel Integrate Block is used to extracted the features from
# every channels also features between channels
# input  : (eg. [batch_size, 36, 500])   groups=3, group_width=12
# output : [batch_size, 48, 500]
# After this block, groups+1, and this block only use once.
# ===================== ChannelInteBlock ==========================
class ChannelInterBlock1st(nn.Module):

    def __init__(
            self,
            in_planes: int,
            input_length: int = 20,  # window length
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ChannelInterBlock1st, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        out_planes = in_planes
        self.input_length = input_length
        self.conv1x1 = convnxn(in_planes, out_planes, kernel_size=1, stride=1, groups=1)
        self.bn = norm_layer(out_planes)
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1x1(x)
        out = self.bn(out)
        out = self.act(out)

        out = torch.cat((identity, out), 1)
        return out

    def get_flops(self):
        flops = 0.0
        # conv1x1
        flops += ((2 * (self.in_planes / self.groups) * 1 - 1) * (
                self.out_planes / self.groups) * self.input_length) * self.groups
        # relu
        flops += self.input_length * self.out_planes
        return flops

    def get_parameters(self):
        parameters = 0.0
        # conv1x1
        parameters += self.in_planes * 1 * self.out_planes
        # bn1
        parameters += 2 * self.out_planes
        return parameters

    def test(self):
        model = DWResBlock(self.in_planes, self.out_planes, kernel_size=3, input_length=500)
        input = torch.ones(size=(1, self.in_planes, self.input_length))
        macs, params = profile(model, inputs=(input,))
        return macs, params


# ===================== ChannelInterBlockN ===========================
# The Channel Integrate Block is used to extracted the features from
# every channels also features between channels
# input  : (eg. [batch_size, 48, 500])   groups=4, group_width=12
# output : [batch_size, 48, 500]
# After this block, groups+1, and this block only use once.
# ===================== ChannelInterBlockN ==========================
class ChannelInterBlockN(nn.Module):

    def __init__(
            self,
            in_planes: int,
            stride: int = 1,
            groups: int = 11,
            input_length: int = 20,  # window length
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ChannelInterBlockN, self).__init__()
        self.input_length = input_length
        self.groups = groups
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.group_width = int(in_planes / self.groups)
        self.conv1x1_1 = convnxn(self.group_width * (self.groups - 1), self.group_width, kernel_size=1,
                                 stride=stride, groups=1)
        self.bn1 = norm_layer(self.group_width)
        self.act = nn.SELU(inplace=True)

        self.conv1x1_2 = convnxn(self.group_width * 2, self.group_width, kernel_size=1, stride=stride,
                                 groups=1)
        self.bn2 = norm_layer(self.group_width)

    def forward(self, x):
        local_information = x[:, :(self.groups - 1) * self.group_width, :]
        global_information = x[:, (self.groups - 1) * self.group_width:, :]

        identity = local_information

        out = self.conv1x1_1(local_information)
        out = self.bn1(out)
        out = self.act(out)

        out = torch.cat((global_information, out), 1)

        out = self.conv1x1_2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = torch.cat((identity, out), 1)

        return out

# =================== EMGNeuralNetwork ======================
#
#
# =================== EMGNeuralNetwork ======================
class EMGNeuralNetwork(nn.Module):

    def __init__(
            self,
            num_classes: int = 52,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.groups = 1
        self.in_planes = 10

        # handle input [b, 10, 20] --> [b, 10, 20]
        self.conv1 = convnxn(10, 20, kernel_size=3, stride=1, groups=10)
        self.bn1 = norm_layer(20)

        self.conv2 = convnxn(20, 40, kernel_size=3, stride=1, groups=10)
        self.bn2 = norm_layer(40)

        self.conv3 = convnxn(40, 80, kernel_size=3, stride=1, groups=10)
        self.bn3 = norm_layer(80)

        self.conv4 = convnxn(80, 160, kernel_size=3, stride=1, groups=10)
        self.bn4 = norm_layer(160)

        self.act = nn.SELU(inplace=True)

        self.adaptiveAvgPool1d = nn.AdaptiveAvgPool1d(50)
        self.decision_layers = nn.Sequential(
            nn.Conv1d(160, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.SELU(),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.Conv1d(64, num_classes, kernel_size=1),
            nn.BatchNorm1d(num_classes),
        )
        self.adaptiveAvgPool1d_2 = nn.AdaptiveAvgPool1d(1)

    def _forward_imp(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act(out)

        out = self.adaptiveAvgPool1d(out)
        out = self.decision_layers(out)
        out = self.adaptiveAvgPool1d_2(out)
        out = torch.flatten(out, 1)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_imp(x)

def _emg_model(
        pretrained: bool = False
):
    model = EMGNeuralNetwork()
    if pretrained:
        pass
    return model


def emgmodel18(pretrained: bool = False, **kwargs: Any) -> EMGNeuralNetwork:
    return _emg_model()
