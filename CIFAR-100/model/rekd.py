import math
import pdb
import torch.nn.functional as F
from torch import nn
import torch

from .mobilenetv2 import mobile_half
from .shufflenetv1 import ShuffleV1
from .shufflenetv2 import ShuffleV2
from .resnet_cifar import build_resnet_backbone, build_resnetx4_backbone
from .vgg import build_vgg_backbone
from .wide_resnet_cifar import wrn

import torch
from torch import nn
import torch.nn.functional as F

from .resnetv2_cifar  import *


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes,
                kernel_size=1, stride=1, padding=0, dilation=1,
                groups=1, with_bn=True, bias=True, init_weight=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes,
                kernel_size, stride, padding, dilation, groups, bias))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
        if init_weight:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(self.conv.weight, a=1)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(self.bn.weight, 1)
                    torch.nn.init.constant_(self.bn.bias, 0)


class VAE(nn.Module):
    def __init__(
        self, in_channels, out_channels, shapes, out_shapes, min_channel, init_weight
    ): 
        super(VAE, self).__init__()

        latten_c = max((out_channels[-1] + in_channels[-1]) // 4, min_channel)

        self.e1 = ConvBN(out_channels[-1], latten_c,
                        kernel_size=1, stride=1, padding=0,
                        with_bn=True, init_weight=init_weight)
        self.e2 = ConvBN(latten_c, latten_c,
                        kernel_size=3, stride=1, padding=1,
                        groups=latten_c, with_bn=True, init_weight=init_weight)
        self.e3 = ConvBN(latten_c, out_channels[-1]*2,
                        kernel_size=1, stride=1, padding=0,
                        with_bn=False, init_weight=init_weight)

        self.d1 = ConvBN(out_channels[-1], latten_c,
                        kernel_size=1, stride=1, padding=0,
                        with_bn=True, init_weight=init_weight)
        self.d2 = ConvBN(latten_c, latten_c,
                        kernel_size=3, stride=1, padding=1,
                        groups=latten_c, with_bn=True, init_weight=init_weight)
        self.d3 = ConvBN(latten_c, out_channels[-1],
                        kernel_size=1, stride=1, padding=0,
                        with_bn=False, init_weight=init_weight)

        self.vbn = nn.BatchNorm2d(out_channels[-1]*2)
        self.last = nn.ReLU(inplace=True)

        self.to('cuda')
    
    def forward(self, x):

        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = self.vbn(self.e3(x))

        mu, logvar = torch.chunk(x,2,dim=1)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = std * eps + mu

        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.last(self.d3(x))

        return x, [mu, std]


class AdapS(nn.Module):
    def __init__(
        self, student, in_channels, out_channels, shapes, out_shapes, min_channel, init_weight
    ): 
        super(AdapS, self).__init__()
        self.in_size = shapes[1]
        self.out_size = out_shapes[1]
        self.refer_c = out_channels[-1]
        self.student = student

        latten_c = max((out_channels[-1] + in_channels[-1]) // 4, min_channel)

        self.a1 = ConvBN(in_channels[-1], latten_c,
                         kernel_size=1, stride=1, padding=0,
                         with_bn=True, init_weight=init_weight)
        self.a2 = ConvBN(latten_c, out_channels[-1],
                         kernel_size=1, stride=1, padding=0,
                         with_bn=False, init_weight=init_weight)

        self.to('cuda')
    
    def forward(self, x):

        student_features, logits = self.student(x, is_feat = True, preact = False)

        fs = student_features[-2]

        if self.in_size > self.out_size:
            fs = F.adaptive_avg_pool2d(
                fs, (self.out_size, self.out_size)
            )
        elif self.in_size < self.out_size:
            fs = F.interpolate(
                fs, (self.out_size, self.out_size), mode='bilinear'
            )

        fa = self.a2(F.relu(self.a1(fs)))

        return fa, logits


def build_re_kd(model, num_classes, teacher = ''):
    out_shapes = None
    init_weight = True
    min_channel = 64
    if 'x4' in model: 
        student = build_resnetx4_backbone(depth = int(model[6:-2]), num_classes = num_classes)
        in_channels = [64,128,256,256]
        out_channels = [64,128,256,256]
        shapes = [1,8,16,32]
    elif 'resnet50' in model:
        student = ResNet50(num_classes = num_classes)
        in_channels = [16,32,64,64]
        out_channels = [16,32,64,64]
        shapes = [1,8,16,32,32]
    elif 'resnet' in model:
        student = build_resnet_backbone(depth = int(model[6:]), num_classes = num_classes)
        in_channels = [16,32,64,64]
        out_channels = [16,32,64,64]
        shapes = [1,8,16,32,32]
    elif 'vgg' in model:
        student = build_vgg_backbone(depth = int(model[3:]), num_classes = num_classes)
        in_channels = [128,256,512,512,512]
        shapes = [1,4,4,8,16]
        if 'resnet50' in teacher:
            out_channels = [256,512,1024,2048,2048]
            out_shapes = [1,4,8,16,32]
        else:
            out_channels = [128,256,512,512,512]
    elif 'mobile' in model:
        init_weight = False
        student = mobile_half(num_classes = num_classes)
        in_channels = [12,16,48,160,1280]
        shapes = [1,2,4,8,16]
        if 'resnet50' in teacher:
            out_channels = [256,512,1024,2048,2048]
            out_shapes = [1,4,8,16,32]
        else:
            out_channels = [128,256,512,512,512]
            out_shapes = [1,4,4,8,16]
    elif 'shuffle' in model:
        init_weight = False
        if 'v1' in model:
            student = ShuffleV1(num_classes = num_classes)
            in_channels = [240,480,960,960]
        elif 'v2' in model:
            student = ShuffleV2(num_classes = num_classes)
            in_channels = [116,232,464,1024]
        else:
            raise NotImplementedError
        shapes = [1,4,8,16]
        if 'wrn' in teacher:
            out_channels = [32,64,128,128]
            out_shapes = [1,8,16,32]
        else:
            out_channels = [64,128,256,256]
            out_shapes = [1,8,16,32]
    elif 'wrn' in model:
        student = wrn(depth=int(model[4:6]), widen_factor=int(model[-1:]), num_classes=num_classes)
        r=int(model[-1:])
        in_channels = [16*r,32*r,64*r,64*r]
        out_channels = [32,64,128,128]
        shapes = [1,8,16,32]
    else:
        assert False

    if out_shapes is None:
        out_shapes = shapes

    backbone = AdapS(
        student=student,
        in_channels=in_channels,
        out_channels=out_channels,
        shapes = shapes,
        out_shapes = out_shapes,
        min_channel = min_channel,
        init_weight=init_weight
    )

    vae = VAE(
        in_channels=in_channels,
        out_channels=out_channels,
        shapes = shapes,
        out_shapes = out_shapes,
        min_channel = min_channel,
        init_weight=init_weight
    )
    return backbone, vae
