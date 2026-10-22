import torch
from torch import nn
import torch.nn.functional as F

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
        self, in_channels, out_channels, init_weight
    ): 
        super(VAE, self).__init__()

        self.s_c = in_channels[-1]
        self.t_c = out_channels[-1]
        self.latten = (self.t_c + self.s_c) // 4

        self.e1 = ConvBN(self.t_c, self.latten,
                        kernel_size=1, stride=1, padding=0,
                        with_bn=True, init_weight=init_weight)
        self.e2 = ConvBN(self.latten, self.latten,
                        kernel_size=3, stride=1, padding=1,
                        groups=self.latten, with_bn=True, init_weight=init_weight)
        self.e3 = ConvBN(self.latten, self.t_c*2,
                        kernel_size=1, stride=1, padding=0,
                        with_bn=False, init_weight=init_weight)

        self.d1 = ConvBN(self.t_c, self.latten,
                        kernel_size=1, stride=1, padding=0,
                        with_bn=True, init_weight=init_weight)
        self.d2 = ConvBN(self.latten, self.latten,
                        kernel_size=3, stride=1, padding=1,
                        groups=self.latten, with_bn=True, init_weight=init_weight)
        self.d3 = ConvBN(self.latten, self.t_c,
                        kernel_size=1, stride=1, padding=0,
                        with_bn=False, init_weight=init_weight)

        self.vbn = nn.BatchNorm2d(self.t_c*2)
        # self.last = nn.ReLU(inplace=True)
        self.last = nn.BatchNorm2d(self.t_c)

        # 1: Identity -> BatchNorm2d
        # 2: latten 4 -> latten 2
        # 3: init_weight False -> True
    
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
    

class AdapL(nn.Module):
    def __init__(
        self, in_channels, out_channels, init_weight
    ): 
        super(AdapL, self).__init__()
        self.s_c = in_channels[-1]
        self.t_c = out_channels[-1]
        self.latten = (self.t_c + self.s_c) // 4

        self.a1 = ConvBN(self.s_c, self.latten,
                         kernel_size=1, stride=1, padding=0,
                         with_bn=True, init_weight=init_weight)
        self.a2 = ConvBN(self.latten, self.t_c,
                         kernel_size=1, stride=1, padding=0,
                         with_bn=False, init_weight=init_weight)
    
    def forward(self, x):
        fa = self.a2(F.relu(self.a1(x)))
        return fa


class ReKD(nn.Module):
    def __init__(
        self, in_channels, out_channels, init_weight
    ): 
        super(ReKD, self).__init__()
        self.s_c = in_channels[-1]
        self.t_c = out_channels[-1]
        self.latten = (self.t_c + self.s_c) // 4

        self.vae = VAE(in_channels, out_channels, init_weight=init_weight)
        self.adapL = AdapL(in_channels, out_channels, init_weight=init_weight)
    
    def forward(self, student_features, teacher_features=None):

        results = []
        if teacher_features is not None:

            rt_features, statistic = self.vae(teacher_features)
            s_features = self.adapL(student_features)
            results = [s_features, rt_features, statistic]

        return results


def build_kd_trans(cfg):
    in_channels = [256,256,256,256,256]
    out_channels = [256,256,256,256,256]
    init_weight = False
    model = ReKD(in_channels, out_channels, init_weight)
    return model