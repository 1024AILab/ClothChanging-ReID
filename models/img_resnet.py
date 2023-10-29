# coding=utf-8
# @FileName:img_resnet.py
# @Time:2023/10/13 
# @Author: CZH

import torchvision
import torch
from torch import nn
from torch.nn import init
from models.utils import pooling
import torch.nn.functional as F


EPSILON = 1e-12

class GEN(nn.Module):
    def __init__(self, in_feat_dim, out_img_dim, config, **kwargs):
        super().__init__()

        self.in_feat_dim = in_feat_dim
        self.out_img_dim = out_img_dim

        self.conv0 = nn.Conv2d(self.in_feat_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
        self.conv4 = nn.Conv2d(32, self.out_img_dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)

        self.up = nn.Upsample(scale_factor=2)

        self.bn = nn.BatchNorm2d(64)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.up(x)
        x = self.conv4(x)
        x = torch.tanh(x)

        return x

class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions, counterfactual=False):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix_raw, dim=-1)

        if counterfactual:
            if self.training:
                fake_att = torch.zeros_like(attentions).uniform_(0, 2)
            else:
                fake_att = torch.ones_like(attentions)
            # mean_feature = features.mean(3).mean(2).view(B, 1, C)
            # counterfactual_feature = mean_feature.expand(B, M, C).contiguous().view(B, -1)
            counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

            counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(
                torch.abs(counterfactual_feature) + EPSILON)

            counterfactual_feature = F.normalize(counterfactual_feature, dim=-1)
            return feature_matrix, counterfactual_feature
        else:
            return feature_matrix

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride = (1, 1)
            resnet50.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.uncloth_dim = config.MODEL.NO_CLOTHES_DIM // 2
        self.contour_dim = config.MODEL.CONTOUR_DIM // 2
        self.cloth_dim = config.MODEL.CLOTHES_DIM // 2

        self.uncloth_net = GEN(in_feat_dim=self.uncloth_dim, out_img_dim=1, config=config)
        self.contour_net = GEN(in_feat_dim=self.contour_dim + self.cloth_dim, out_img_dim=1, config=config)
        self.cloth_net = GEN(in_feat_dim=self.cloth_dim, out_img_dim=1, config=config)
        dim = 1024
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction1 = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, dim, kernel_size=1)
        )

        self.se = SELayer(1024)
        self.cal = BAP(pool='GAP')
        self.bnse = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = self.base(x)
        se = self.bnse(self.se(x))
        bap1, bap2 = self.cal(x, se, True)
        bap1 = bap1 - bap2
        channel_map = self.channel_interaction1(x)
        spatial_map = self.spatial_interaction1(bap1)
        channel_gate = se * torch.sigmoid(spatial_map)
        spatial_gate = bap1 * torch.sigmoid(channel_map)
        x = channel_gate + spatial_gate

        x_ori = x
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        f_unclo = x_ori[:, 0:self.uncloth_dim, :, :]
        f_cont = x_ori[:, self.uncloth_dim:self.uncloth_dim + self.contour_dim + self.cloth_dim, :, :]
        f_clo = x_ori[:, self.uncloth_dim + self.contour_dim:self.uncloth_dim + self.contour_dim + self.cloth_dim, :, :]

        unclo_img = self.uncloth_net(f_unclo)
        cont_img = self.contour_net(f_cont)
        clo_img = self.cloth_net(f_clo)
        # unclo_img 没有衣服
        # clo_img 有衣服
        # cont_img 轮廓图
        return (f, unclo_img, cont_img, clo_img)