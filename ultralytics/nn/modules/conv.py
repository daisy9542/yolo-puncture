# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "ECA",
    "ELA",
    "CAA",
    "EMA",
    "ACS",
    "MCA",
    "SRM",
    "SE",
    "CA",
    "SA",
    "DIYv26",
    "DIYv27",
    "DIYv28",
    "ACAM",
    "DCIM",
    "LSAM",
    "MSTM",
    "CPEM",
    "FSFM",
    "CAPM",
    "LHAM",
    "EEM",
    "NAFM",
    "ISRM",
    "SSRM",
    "CSRM",
    "GNSRM",
    "MSRM",
    "NLSRM",
    "ResSRM",
    "FSSRM",
    "GCSRM",
    "DCSRM",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    
    default_act = nn.SiLU()  # default activation
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv
    
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))
    
    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
    
    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]: i[0] + 1, i[1]: i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    
    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)
    
    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""
    
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""
    
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    
    default_act = nn.SiLU()  # default activation
    
    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))
    
    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)
    
    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""
    
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
    
    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    
    default_act = nn.SiLU()  # default activation
    
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
    
    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))
    
    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)
    
    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAM(nn.Module):
    
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    
    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class ECA(nn.Module):
    """Constructs an ECA module.

    Args:
        k_size: Adaptive selection of kernel size
    """
    
    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class ELA(nn.Module):
    
    def __init__(self, in_channels, phi):
        """
        ELA-T 和 ELA-B 设计为轻量级，非常适合网络层数较少或轻量级网络的 CNN 架构
        ELA-B 和 ELA-S 在具有更深结构的网络上表现最佳
        ELA-L 特别适合大型网络

        Args:
            in_channels: 输入特征图的通道数
            phi: 卷积核大小和组数的选择，phi='T'、'B'、'S'、'L'
        """
        super(ELA, self).__init__()
        kernel_size = {'T': 5, 'B': 7, 'S': 5, 'L': 7}[phi]
        groups = {'T': in_channels, 'B': in_channels, 'S': in_channels // 8, 'L': in_channels // 8}[phi]
        num_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}[phi]
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=pad, groups=groups, bias=False)
        self.GN = nn.GroupNorm(num_groups, in_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        x_h = torch.mean(x, 3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, 2, keepdim=True).view(b, c, w)
        x_h = self.conv1(x_h)
        x_w = self.conv1(x_w)
        x_h = self.sigmoid(self.GN(x_h)).view(b, c, h, 1)
        x_w = self.sigmoid(self.GN(x_w)).view(b, c, 1, w)
        return x_h * x_w * x


class CAA(BaseModule):
    """Context Anchor Attention"""
    
    def __init__(
            self,
            channels,
            h_kernel_size=11,
            v_kernel_size=11,
            norm_cfg=None,
            act_cfg=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        if act_cfg is None:
            act_cfg = dict(type='SiLU')
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg
                                )
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None
                                 )
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None
                                 )
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg
                                )
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ACS(nn.Module):
    def __init__(self, in_channels, groups=2):
        super(ACS, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.group_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=groups)
    
    def forward(self, x):
        u = x
        x = self.avgpool(x)
        x = self.conv(x)
        x = F.softmax(x, dim=1)
        out1 = u * x
        out2 = self.group_conv(u)
        out = out1 + F.interpolate(out2, size=out1.shape[2:], mode='bilinear', align_corners=False)
        return out


class StdPool(nn.Module):
    def __init__(self):
        super(StdPool, self).__init__()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        std = x.view(b, c, -1).std(dim=2, keepdim=True)
        std = std.reshape(b, c, 1, 1)
        
        return std


class MCAGate(nn.Module):
    def __init__(self, k_size, pool_types=['avg', 'std']):
        """Constructs a MCAGate module.
        Args:
            k_size: kernel size
            pool_types: pooling type. 'avg': average pooling, 'max': max pooling, 'std': standard deviation pooling.
        """
        super(MCAGate, self).__init__()
        
        self.pools = nn.ModuleList([])
        for pool_type in pool_types:
            if pool_type == 'avg':
                self.pools.append(nn.AdaptiveAvgPool2d(1))
            elif pool_type == 'max':
                self.pools.append(nn.AdaptiveMaxPool2d(1))
            elif pool_type == 'std':
                self.pools.append(StdPool())
            else:
                raise NotImplementedError
        
        self.conv = nn.Conv2d(1, 1, kernel_size=(1, k_size), stride=1, padding=(0, (k_size - 1) // 2), bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.weight = nn.Parameter(torch.rand(2))
    
    def forward(self, x):
        feats = [pool(x) for pool in self.pools]
        
        if len(feats) == 1:
            out = feats[0]
        elif len(feats) == 2:
            weight = torch.sigmoid(self.weight)
            out = 1 / 2 * (feats[0] + feats[1]) + weight[0] * feats[0] + weight[1] * feats[1]
        else:
            assert False, "Feature Extraction Exception!"
        
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv(out)
        out = out.permute(0, 3, 2, 1).contiguous()
        
        out = self.sigmoid(out)
        out = out.expand_as(x)
        
        return x * out


class MCA(nn.Module):
    def __init__(self, inp, no_spatial=False):
        """Constructs an MCA module.
        Args:
            inp: Number of channels of the input feature maps
            no_spatial: whether to build channel dimension interactions
        """
        super(MCA, self).__init__()
        
        lambd = 1.5
        gamma = 1
        temp = round(abs((math.log2(inp) - gamma) / lambd))
        kernel = temp if temp % 2 else temp - 1
        
        self.h_cw = MCAGate(3)
        self.w_hc = MCAGate(3)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.c_hw = MCAGate(kernel)
    
    def forward(self, x):
        x_h = x.permute(0, 2, 1, 3).contiguous()
        x_h = self.h_cw(x_h)
        x_h = x_h.permute(0, 2, 1, 3).contiguous()
        
        x_w = x.permute(0, 3, 2, 1).contiguous()
        x_w = self.w_hc(x_w)
        x_w = x_w.permute(0, 3, 2, 1).contiguous()
        
        if not self.no_spatial:
            x_c = self.c_hw(x)
            x_out = 1 / 3 * (x_c + x_h + x_w)
        else:
            x_out = 1 / 2 * (x_h + x_w)
        
        return x_out


class SRM(nn.Module):
    def __init__(self, channel):
        super(SRM, self).__init__()
        
        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        
        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()
        
        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()
        
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        
        #############
        # 如果空间维度为 (1, 1)，则跳过 BatchNorm2d
        if z.size(2) == 1 and z.size(3) == 1:
            z_hat = z
        else:
            #############
            z_hat = self.bn(z)
        g = self.activation(z_hat)
        
        return g
    
    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)
        
        # B x C x 1 x 1
        g = self._style_integration(t)
        
        return x * g


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.activation = nn.Sigmoid()
        
        self.reduction = reduction
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // self.reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // self.reduction, channel),
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_y = self.avgpool(x).view(b, c)
        
        gate = self.fc(avg_y).view(b, c, 1, 1)
        gate = self.activation(gate)
        
        return x * gate


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    
    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        
        return out


class SA(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """
    
    def __init__(self, channel, groups=32):
        super(SA, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
    
    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        
        # flatten
        x = x.reshape(b, -1, h, w)
        
        return x
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        
        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        
        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)
        
        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        
        out = self.channel_shuffle(out, 2)
        return out


# v7 v8
# class DIY(nn.Module):
#     """Efficient Layer-Context Attention"""
#
#     def __init__(self, in_channels):
#         super(DIY, self).__init__()
#         self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         x_h = self.avg_pool_h(x)  # [b, c, h, 1]
#         x_w = self.avg_pool_w(x)  # [b, c, 1, w]
#         x_h = self.fc(x_h)  # [b, c, h, 1]
#         x_w = self.fc(x_w)  # [b, c, 1, w]
#         x2 = self.sigmoid(x_h + x_w)  # [b, c, h, w]
#         x3 = self.act(self.conv1(x))  # [b, c, h, w]
#         x4 = x2 * x3
#         out = x4 * x
#         return out


# v9
# class DIY(nn.Module):
#     """Efficient Layer-Context Attention"""
#
#     def __init__(self, in_channels, h_kernel_size=11, v_kernel_size=11, norm_cfg=None, act_cfg=None):
#         """
#         Efficient Layer-Context Attention Module (DIY)
#
#         该模块结合了 ELA 和 CAA 模块的功能，通过捕获高效的层级和上下文注意力来增强特征表示。
#
#         Args:
#             in_channels (int): 输入特征图的通道数。
#             h_kernel_size (int): CAA 模块中水平卷积的核大小。
#             v_kernel_size (int): CAA 模块中垂直卷积的核大小。
#             norm_cfg (dict): 归一化层的配置。
#             act_cfg (dict): 激活函数的配置。
#         """
#         super(DIY, self).__init__()
#
#         # ELA 模块的组件，phi 固定为 'T'
#         kernel_size = 5
#         groups = in_channels  # 深度可分离卷积
#         pad = kernel_size // 2
#         num_groups = min(32, in_channels)  # 确保 num_groups 不超过 in_channels
#
#         self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=pad, groups=groups, bias=False)
#         self.GN = nn.GroupNorm(num_groups, in_channels)
#         self.sigmoid = nn.Sigmoid()
#
#         # 如果未提供 norm_cfg 和 act_cfg，则使用默认值
#         norm_cfg = {'type': 'BN', 'momentum': 0.03, 'eps': 0.001}
#         act_cfg = {'type': 'SiLU'}
#
#         # 定义归一化层
#         def get_norm_layer(norm_cfg, num_features):
#             norm_type = norm_cfg.get('type', 'BN')
#             if norm_type == 'BN':
#                 return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1),
#                                       eps=norm_cfg.get('eps', 1e-5))
#             elif norm_type == 'GN':
#                 num_groups = norm_cfg.get('num_groups', 32)
#                 return nn.GroupNorm(num_groups, num_features, eps=norm_cfg.get('eps', 1e-5))
#             else:
#                 raise ValueError(f"Unsupported norm type: {norm_type}")
#
#         # 定义激活函数
#         def get_activation(act_cfg):
#             act_type = act_cfg.get('type', 'ReLU')
#             if act_type == 'ReLU':
#                 return nn.ReLU(inplace=True)
#             elif act_type == 'SiLU':
#                 return nn.SiLU(inplace=True)
#             elif act_type == 'LeakyReLU':
#                 negative_slope = act_cfg.get('negative_slope', 0.01)
#                 return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
#             else:
#                 raise ValueError(f"Unsupported activation type: {act_type}")
#
#         # CAA 模块的组件
#         self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
#
#         # CAA 模块的第一个卷积层
#         self.conv1_caa = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
#             get_norm_layer(norm_cfg, in_channels),
#             get_activation(act_cfg)
#         )
#
#         # 水平卷积层
#         self.h_conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=(1, h_kernel_size), stride=1,
#             padding=(0, h_kernel_size // 2), groups=in_channels, bias=False
#         )
#
#         # 垂直卷积层
#         self.v_conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=(v_kernel_size, 1), stride=1,
#             padding=(v_kernel_size // 2, 0), groups=in_channels, bias=False
#         )
#
#         # CAA 模块的第二个卷积层
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
#             get_norm_layer(norm_cfg, in_channels),
#             get_activation(act_cfg)
#         )
#
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # ELA 模块的前向传播
#         x_h = torch.mean(x, dim=3, keepdim=True)  # 在宽度方向上平均
#         x_w = torch.mean(x, dim=2, keepdim=True)  # 在高度方向上平均
#         x_h = x_h.view(b, c, h)
#         x_w = x_w.view(b, c, w)
#
#         x_h = self.conv1(x_h)
#         x_w = self.conv1(x_w)
#         x_h = self.GN(x_h)
#         x_w = self.GN(x_w)
#         x_h = self.sigmoid(x_h).view(b, c, h, 1)
#         x_w = self.sigmoid(x_w).view(b, c, 1, w)
#         attention_ela = x_h * x_w  # ELA 的注意力图
#
#         # CAA 模块的前向传播
#         x_pool = self.avg_pool(x)
#         x_caa = self.conv1_caa(x_pool)
#         x_caa = self.h_conv(x_caa)
#         x_caa = self.v_conv(x_caa)
#         x_caa = self.conv2(x_caa)
#         attention_caa = self.act(x_caa)
#
#         # 将两个注意力图相乘并应用到输入上
#         combined_attention = attention_ela * attention_caa
#         out = combined_attention * x
#         return out


# v11
# class DIY(BaseModule):
#     """Efficient Layer-Context Attention"""
#
#     def __init__(
#             self,
#             channels,
#             h_kernel_size=11,
#             v_kernel_size=11,
#             groups=2,
#     ):
#         super().__init__()
#         norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
#         act_cfg = dict(type='SiLU')
#         self.avg_pool = nn.AvgPool2d(7, 1, 3)
#         self.conv1 = ConvModule(channels, channels, 1, 1, 0,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg
#                                 )
#         self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
#                                  (0, h_kernel_size // 2), groups=channels,
#                                  norm_cfg=None, act_cfg=None
#                                  )
#         self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
#                                  (v_kernel_size // 2, 0), groups=channels,
#                                  norm_cfg=None, act_cfg=None
#                                  )
#         self.conv2 = ConvModule(channels, channels, 1, 1, 0,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg
#                                 )
#         self.act = nn.Sigmoid()
#         self.group_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=groups)
#
#     def forward(self, x):
#         out1 = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
#         out2 = self.group_conv(x)
#         out = out1 + F.interpolate(out2, size=out1.shape[2:], mode='bilinear', align_corners=False)
#         return out


class DIYv26(nn.Module):
    def __init__(self, channel, reduction=16, no_spatial=False):
        super(DIYv26, self).__init__()
        self.se = SE(channel, reduction)
        self.mca = MCA(channel, no_spatial)
        self.conv_fuse = nn.Conv2d(channel * 2, channel, kernel_size=1)
    
    def forward(self, x):
        se_out = self.se(x)
        mca_out = self.mca(x)
        concat_out = torch.cat([se_out, mca_out], dim=1)
        out = self.conv_fuse(concat_out)
        return out


class DIYv27(nn.Module):
    def __init__(self, channels, factor=32):
        super(DIYv27, self).__init__()
        self.ema = EMA(channels, factor)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        ema_out = self.ema(x)
        out = self.conv(ema_out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DIYv28(nn.Module):
    def __init__(self, channels, reduction=16):
        super(DIYv28, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        channel_attn = self.fc(torch.cat([avg_out, max_out], dim=1))
        channel_attn = self.sigmoid(channel_attn)
        x = x * channel_attn
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        out = x * spatial_attn
        return out


########## 以下均为全新创造模块 ##########

class ISRM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ISRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 通道注意力机制（类似于SE模块）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        # 原始的SRM处理
        t = self._style_pooling(x)
        g = self._style_integration(t)
        srm_out = x * g
        
        # 通道注意力处理
        ca = self.channel_attn(srm_out)
        out = srm_out * ca
        return out


class SSRM(nn.Module):
    def __init__(self, channel):
        super(SSRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 空间注意力机制
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        channel_var = x.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        # 原始的SRM处理
        t = self._style_pooling(x)
        g = self._style_integration(t)
        srm_out = x * g
        
        # 空间注意力处理
        avg_out = torch.mean(srm_out, dim=1, keepdim=True)
        max_out, _ = torch.max(srm_out, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        out = srm_out * sa
        return out


class CSRM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CSRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 通道和空间注意力机制
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        channel_var = x.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        # 原始的SRM处理
        t = self._style_pooling(x)
        g = self._style_integration(t)
        srm_out = x * g
        
        # 通道注意力
        ca = self.channel_attn(srm_out)
        x_ca = srm_out * ca
        
        # 空间注意力
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        out = x_ca * sa
        return out


class GNSRM(nn.Module):
    def __init__(self, channel, groups=32):
        super(GNSRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 组归一化层
        self.gn = nn.GroupNorm(groups, channel)
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        channel_var = x.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        # 原始的SRM处理
        t = self._style_pooling(x)
        g = self._style_integration(t)
        srm_out = x * g
        
        # 组归一化处理
        out = self.gn(srm_out)
        return out


class MSRM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MSRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 多尺度卷积
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channel, channel, kernel_size=5, padding=2)
        self.fuse_conv = nn.Conv2d(channel * 3, channel, kernel_size=1)
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        channel_var = x.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        # 原始的SRM处理
        t = self._style_pooling(x)
        g = self._style_integration(t)
        srm_out = x * g
        
        # 多尺度卷积
        x1 = self.conv1(srm_out)
        x3 = self.conv3(srm_out)
        x5 = self.conv5(srm_out)
        multi_scale = torch.cat([x1, x3, x5], dim=1)
        out = self.fuse_conv(multi_scale)
        return out


class NLSRM(nn.Module):
    def __init__(self, channel):
        super(NLSRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 非局部注意力机制
        self.theta = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.phi = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.g = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.out_conv = nn.Conv2d(channel // 2, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        channel_var = x.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        # 原始的SRM处理
        t = self._style_pooling(x)
        srm_g = self._style_integration(t)
        srm_out = x * srm_g
        
        # 非局部注意力
        batch_size, C, H, W = srm_out.size()
        theta = self.theta(srm_out).view(batch_size, C // 2, -1)
        phi = self.phi(srm_out).view(batch_size, C // 2, -1)
        g = self.g(srm_out).view(batch_size, C // 2, -1)
        
        attention = torch.bmm(theta.permute(0, 2, 1), phi)
        attention = self.softmax(attention)
        
        nl_out = torch.bmm(g, attention.permute(0, 2, 1))
        nl_out = nl_out.view(batch_size, C // 2, H, W)
        nl_out = self.out_conv(nl_out)
        
        out = srm_out + nl_out
        return out


class ResSRM(nn.Module):
    def __init__(self, channel):
        super(ResSRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 残差连接中的卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        )
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        channel_var = x.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        identity = x
        # 原始的SRM处理
        t = self._style_pooling(x)
        g = self._style_integration(t)
        srm_out = x * g
        
        # 残差连接
        res_out = self.conv(srm_out)
        out = srm_out + res_out + identity
        return out


class FSSRM(nn.Module):
    def __init__(self, channel):
        super(FSSRM, self).__init__()
        # 风格信息提取
        self.cfc_style = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc_style.data.fill_(0)
        
        # 空间信息提取
        self.cfc_spatial = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc_spatial.data.fill_(0)
        
        self.activation = nn.Sigmoid()
        
        setattr(self.cfc_style, 'srm_param', True)
        setattr(self.cfc_spatial, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        # 风格信息
        channel_mean = x.view(N, C, -1).mean(dim=2)
        channel_var = x.view(N, C, -1).var(dim=2) + eps
        channel_std = channel_var.sqrt()
        t_style = torch.cat((channel_mean, channel_std), dim=1)
        
        # 空间信息
        spatial_mean = x.mean(dim=1).view(N, -1)
        spatial_var = x.var(dim=1).view(N, -1) + eps
        spatial_std = spatial_var.sqrt()
        t_spatial = torch.cat((spatial_mean, spatial_std), dim=1)
        
        return t_style, t_spatial
    
    def _style_integration(self, t_style, t_spatial):
        # 风格信息融合
        z_style = t_style * self.cfc_style[None, :, :]  # B x C x 2
        z_style = torch.sum(z_style, dim=2)[:, :, None, None]  # B x C x 1 x 1
        
        # 空间信息融合
        z_spatial = t_spatial * self.cfc_spatial[None, :, :]  # B x HW x 2
        z_spatial = torch.sum(z_spatial, dim=2)[:, :, None, None]  # B x HW x 1 x 1
        
        # 合并
        z = z_style + z_spatial
        g = self.activation(z)
        return g
    
    def forward(self, x):
        # 风格和空间信息提取
        t_style, t_spatial = self._style_pooling(x)
        g = self._style_integration(t_style, t_spatial)
        out = x * g
        return out


class GCSRM(nn.Module):
    def __init__(self, channel, groups=32):
        super(GCSRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 组卷积层
        self.group_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=groups)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        channel_var = x.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        # 原始的SRM处理
        t = self._style_pooling(x)
        g = self._style_integration(t)
        srm_out = x * g
        
        # 组卷积处理
        gc_out = self.group_conv(srm_out)
        gc_out = self.bn(gc_out)
        out = self.relu(gc_out)
        return out


class DCSRM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3):
        super(DCSRM, self).__init__()
        # 原始的SRM部分
        self.cfc = nn.Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        
        # 动态卷积权重生成器
        self.dynamic_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel * kernel_size * kernel_size, kernel_size=1)
        )
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        setattr(self.cfc, 'srm_param', True)
    
    def _style_pooling(self, x, eps=1e-5):
        N, C, H, W = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        channel_var = x.view(N, C, -1).var(dim=2).view(N, C, 1, 1) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1
        g = self.activation(z)
        return g
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # 原始的SRM处理
        t = self._style_pooling(x)
        g = self._style_integration(t)
        srm_out = x * g
        
        # 生成动态卷积核
        dynamic_weights = self.dynamic_weight(srm_out)
        dynamic_weights = dynamic_weights.view(batch_size * channels, 1, self.kernel_size, self.kernel_size)
        
        # 对每个通道进行动态卷积
        srm_out = srm_out.view(1, batch_size * channels, height, width)
        out = F.conv2d(srm_out, dynamic_weights, padding=self.padding, groups=batch_size * channels)
        out = out.view(batch_size, channels, height, width)
        return out


class ACAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ACAM, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.global_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fusion_conv = nn.Conv2d(in_channels // reduction * 2, in_channels, kernel_size=1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # 全局上下文特征
        global_context = self.global_pool(x)
        global_context = self.global_conv(global_context)
        
        # 局部上下文特征
        local_context = self.local_conv(x)
        
        # 特征融合
        combined = torch.cat([local_context, global_context.expand_as(local_context)], dim=1)
        out = self.fusion_conv(combined)
        attn = self.activation(out)
        return x * attn


class DCIM(nn.Module):
    def __init__(self, in_channels, num_experts=4, reduction=16):
        super(DCIM, self).__init__()
        self.num_experts = num_experts
        self.fc = nn.Linear(in_channels, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
            ) for _ in range(num_experts)
        ]
        )
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        weights = self.softmax(self.fc(y))  # B x num_experts
        
        out = 0
        for i, expert in enumerate(self.experts):
            out += weights[:, i:i + 1].view(b, 1, 1, 1) * expert(x)
        attn = self.activation(out)
        return x * attn


class LSAM(nn.Module):
    def __init__(self, in_channels, window_size=7):
        super(LSAM, self).__init__()
        self.window_size = window_size
        self.conv_q = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        b, c, h, w = x.size()
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H, W = x.size()
        
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        
        q = q.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)
        k = k.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)
        v = v.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)
        
        q = q.contiguous().view(b, -1, self.window_size * self.window_size)
        k = k.contiguous().view(b, -1, self.window_size * self.window_size)
        v = v.contiguous().view(b, -1, self.window_size * self.window_size)
        
        attn = self.softmax(torch.bmm(q.transpose(1, 2), k))
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, H, W)
        return out[:, :, :h, :w]


class MSTM(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MSTM, self).__init__()
        self.num_heads = num_heads
        self.scale = (in_channels // num_heads)**-0.5
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.size()
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        out = self.proj(out)
        return out


class CPEM(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(CPEM, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.activation = nn.GELU()
        self.conv = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.proj(x)
        x = x + self.pos_embed
        x = self.activation(x)
        x = self.conv(x)
        return x


class FSFM(nn.Module):
    def __init__(self, in_channels):
        super(FSFM, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        depthwise_out = self.depthwise_conv(x)
        pointwise_out = self.pointwise_conv(x)
        combined = torch.cat([depthwise_out, pointwise_out], dim=1)
        out = self.fusion_conv(combined)
        out = self.activation(out)
        return out


class CAPM(nn.Module):
    def __init__(self, in_channels, scales=[1, 3, 5]):
        super(CAPM, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=s, padding=s // 2)
            for s in scales
        ]
        )
        self.fusion_conv = nn.Conv2d(in_channels * len(scales), in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        combined = torch.cat(features, dim=1)
        out = self.fusion_conv(combined)
        out = self.activation(out)
        return out


class LHAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(LHAM, self).__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attn(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        out = x * sa
        return out


class EEM(nn.Module):
    def __init__(self, in_channels):
        super(EEM, self).__init__()
        self.edge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.edge_conv.weight.data = self.get_sobel_kernel(in_channels)
        self.edge_conv.bias.data.zero_()
        self.edge_conv.requires_grad = False
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
    
    def get_sobel_kernel(self, channels):
        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32
                              ).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel
    
    def forward(self, x):
        edge = self.edge_conv(x)
        combined = torch.cat([x, edge], dim=1)
        out = self.conv(combined)
        out = self.activation(out)
        return out


class NAFM(nn.Module):
    def __init__(self, in_channels):
        super(NAFM, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.prelu = nn.PReLU()
        self.swish = nn.SiLU()
    
    def forward(self, x):
        out1 = self.prelu(self.conv(x))
        out2 = self.swish(self.conv(x))
        out = out1 + out2
        return out
