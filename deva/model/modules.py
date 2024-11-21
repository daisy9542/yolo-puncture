"""
modules.py - This file stores low-level network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

from typing import List, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F

from deva.model.group_modules import *
from deva.model.cbam import CBAM


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        out_f = self.conv1(F.relu(f))
        out_f = self.conv2(F.relu(out_f))

        if self.downsample is not None:
            f = self.downsample(f)

        return out_f + f


class FeatureFusionBlock(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super().__init__()

        self.block1 = ResBlock(in_dim, mid_dim)
        self.attention = CBAM(mid_dim)
        self.block2 = ResBlock(mid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


class KeyProjection(nn.Module):
    def __init__(self, in_dim: int, keydim: int):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x: torch.Tensor, *, need_s: bool,
                need_e: bool) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class MaskUpsampleBlock(nn.Module):
    def __init__(self, up_dim: int, out_dim: int, scale_factor: int = 2):
        super().__init__()
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(up_dim, out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f: torch.Tensor, up_g: torch.Tensor) -> torch.Tensor:
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class DecoderFeatureProcessor(nn.Module):
    def __init__(self, decoder_dims: List[int], out_dims: List[int]):
        super().__init__()
        self.transforms = nn.ModuleList([
            nn.Conv2d(d_dim, p_dim, kernel_size=1) for d_dim, p_dim in zip(decoder_dims, out_dims)
        ])

    def forward(self, multi_scale_features: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        outputs = [func(x) for x, func in zip(multi_scale_features, self.transforms)]
        return outputs


class LinearPredictor(nn.Module):
    def __init__(self, in_dim: int, pred_dim: int):
        super().__init__()
        self.projection = GConv2D(in_dim, pred_dim + 1, kernel_size=1)

    def forward(self, im_feat: torch.Tensor, pred_feat: torch.Tensor) -> torch.Tensor:
        num_objects = pred_feat.shape[1]
        parameters = self.projection(pred_feat)

        im_feat = im_feat.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)
        x = (im_feat * parameters[:, :, :-1]).sum(dim=2, keepdim=True) + parameters[:, :, -1:]
        return x


class SensoryUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims: List[int], mid_dim: int, sensory_dim: int):
        super().__init__()
        self.sensory_dim = sensory_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU,
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :, :self.sensory_dim])
        update_gate = torch.sigmoid(values[:, :, self.sensory_dim:self.sensory_dim * 2])
        new_value = torch.tanh(values[:, :, self.sensory_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h


class SensoryDeepUpdater(nn.Module):
    def __init__(self, f_dim: int, sensory_dim: int):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.transform = GConv2D(f_dim + sensory_dim, sensory_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, f: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        values = self.transform(torch.cat([f, h], dim=2))
        forget_gate = torch.sigmoid(values[:, :, :self.sensory_dim])
        update_gate = torch.sigmoid(values[:, :, self.sensory_dim:self.sensory_dim * 2])
        new_value = torch.tanh(values[:, :, self.sensory_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h


class MemoryReinforceModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(MemoryReinforceModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_out(x)
        return x


def reinforce_features(yolo_features, prev_frame_mask):
    print("---------------reinforce_features----------------")
    for f in yolo_features:
        print("yolo_feature", f.shape)
    print("prev_frame_mask: ", prev_frame_mask.shape)
    # 上采样特征图到80×80
    [feature_1, feature_2, feature_3] = yolo_features
    feature_1_upsampled = feature_1  # 已经是80×80
    feature_2_upsampled = F.interpolate(feature_2, size=feature_1.shape[-2:], mode='bilinear', align_corners=False)
    feature_3_upsampled = F.interpolate(feature_3, size=feature_1.shape[-2:], mode='bilinear', align_corners=False)
    
    # 调整通道数
    conv1x1_c1 = nn.Conv2d(feature_1.shape[1], 256, kernel_size=1)
    conv1x1_c2 = nn.Conv2d(feature_2.shape[1], 256, kernel_size=1)
    conv1x1_c3 = nn.Conv2d(feature_3.shape[1], 256, kernel_size=1)
    
    feature_1_adjusted = conv1x1_c1(feature_1_upsampled)
    feature_2_adjusted = conv1x1_c2(feature_2_upsampled)
    feature_3_adjusted = conv1x1_c3(feature_3_upsampled)
    
    # 融合特征图
    fused_feature = torch.cat([feature_1_adjusted, feature_2_adjusted, feature_3_adjusted], dim=1)  # 通道数768
    
    # 调整prev_frame_mask尺寸并扩展通道数
    prev_mask_resized = F.interpolate(prev_frame_mask, size=feature_1.shape[-2:], mode='bilinear', align_corners=False)
    mask_conv = nn.Conv2d(1, 768, kernel_size=1)
    prev_mask_adjusted = mask_conv(prev_mask_resized)
    
    # 拼接特征图和掩码
    concatenated_feature = torch.cat([fused_feature, prev_mask_adjusted], dim=1)  # 通道数1536
    
    # 应用记忆强化模块
    memory_module = MemoryReinforceModule(in_channels=1536, mid_channels=512, out_channels=256)
    reinforced_feature = memory_module(concatenated_feature)
    
    return reinforced_feature
