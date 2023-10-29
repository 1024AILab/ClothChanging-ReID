# coding=utf-8
# @FileName:OSA.py
# @Time:2023/10/20 
# @Author: CZH
# 来源Adaptive Multilayer Perceptual Attention Network for Facial Expression Recognition

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(),
                                 nn.Linear(gate_channels, gate_channels // reduction_ratio),
                                 nn.ReLU(),
                                 nn.Linear(gate_channels // reduction_ratio, gate_channels))
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
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class OcclusionGate(nn.Module):
    def __init__(self, gate_channels):
        super(OcclusionGate, self).__init__()

        kernel_size = 3
        self.spatial = BasicConv(gate_channels, int(gate_channels / 2), kernel_size, stride=1, padding=1, relu=True)
        self.spatial1 = BasicConv(int(gate_channels / 2), int(gate_channels / 2), kernel_size, stride=2, padding=1,
                                  relu=True)
        self.fc2 = nn.Linear(int(gate_channels / 2), 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # occlusion attention
        x_out = self.spatial(x)
        x_out = self.spatial1(x_out)
        x_out = self.avgpool(x_out)
        x_out = torch.flatten(x_out, 1)
        x_out = self.fc2(x_out)
        scale = torch.sigmoid(x_out)
        # x : fc
        x_out1 = self.avgpool(x)
        x_out1 = torch.flatten(x_out1, 1)

        return x_out1 * scale


class sigatten(nn.Module):
    def __init__(self, gate_channels):
        super(sigatten, self).__init__()
        self.fc2 = nn.Linear(gate_channels, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # occlusion attention

        x_out = self.avgpool(x)
        x_out = torch.flatten(x_out, 1)
        x_out = self.fc2(x_out)
        scale = torch.sigmoid(x_out)
        scale = scale.unsqueeze(2)
        scale = scale.unsqueeze(3)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)

        return x_out


class CBAM_O(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM_O, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        # self.OcclusionGate =OcclusionGate(gate_channels)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out


class CBAM_OL(nn.Module):
    def __init__(self, gate_channels):
        super(CBAM_OL, self).__init__()
        self.sigatten = sigatten(gate_channels)
        # self.OcclusionGate =OcclusionGate(gate_channels)

    def forward(self, x):
        x_out = self.sigatten(x)
        return x_out


def dw_conv3x3(in_channels, out_channels, module_name, postfix,
               stride=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/dw_conv3x3'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=out_channels,
                   bias=False)),
        ('{}_{}/pw_conv1x1'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=1,
                   stride=1,
                   padding=0,
                   groups=1,
                   bias=False)),
        ('{}_{}/pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ('{}_{}/pw_cbam'.format(module_name, postfix), CBAM_OL(out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),

    ]


def conv3x3_v1(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        ('{}_{}/pw_cbam'.format(module_name, postfix), CBAM_OL(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv3x3(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        ('{}_{}/pw_cbam'.format(module_name, postfix), CBAM_OL(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1(
        in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):
    def __init__(
            self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, identity=True, depthwise=True
    ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.drop2d = nn.Dropout2d(0.5)
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True

            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1(in_channel, stage_ch,
                                    "{}_reduction".format(module_name), "0")))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(
                    nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(
                    nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i)))
                )
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat"))
        )

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat
        xt = self.drop2d(xt)
        return xt


if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('logs')
    model = _OSA_module(in_ch=1024, stage_ch=1024, concat_ch=1024, layer_per_block=6, module_name=3)
    input = torch.randn(32, 1024, 24, 12)
    # output = model(input)
    writer.add_graph(model, input)
    # print(output.shape)
