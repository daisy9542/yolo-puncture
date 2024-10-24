# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
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
    "DIY",
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
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""
    
    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""
    
    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


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
class DIY(BaseModule):
    """Efficient Layer-Context Attention"""
    
    def __init__(
            self,
            channels,
            h_kernel_size=11,
            v_kernel_size=11,
            groups=2,
    ):
        super().__init__()
        norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
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
        self.group_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, groups=groups)
    
    def forward(self, x):
        out1 = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        out2 = self.group_conv(x)
        out = out1 + F.interpolate(out2, size=out1.shape[2:], mode='bilinear', align_corners=False)
        return out
