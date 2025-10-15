"""
modules.py - This file stores low-level network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""
from modulefinder import Module
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
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1 / 2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1 / 4))
        
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


class FeatureReinforceModule(nn.Module):
    def __init__(self,
                 w: int = 1,
                 r: int = 1):
        """
        Fusion module to combine YOLO multi-stage features with XMem memory features.

        Parameters:
            w (int): Widen factor from YOLO.
            r (int): Ratio from YOLO.
        """
        super().__init__()
        self.w = w
        self.r = r
        
        self.conv_yolo1 = nn.Conv2d(256 * w, 512, kernel_size=1)
        self.conv_yolo2 = nn.Conv2d(512 * w, 512, kernel_size=1)
        self.conv_yolo3 = nn.Conv2d(512 * w * r, 512, kernel_size=1)
        
        # 用于将三路 YOLO 特征拼接后的通道数再压缩到 512
        self.fusion_conv = nn.Conv2d(512 * 3, 512, kernel_size=1)
        
        # Gating 卷积 (将 YOLO_fused 与 XMem 拼接后做 1x1 卷积得到门控系数)
        self.gate_conv = nn.Conv2d(512 * 2, 1, kernel_size=1)
        
        # 最终融合后再进行一次 3x3 卷积精细化特征
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
    
    def forward(self,
                yolo_features: List[torch.Tensor],
                masks: torch.Tensor,
                ms_features: List[torch.Tensor]):
        """
        batch_size = 1
        
        Args:
            yolo_features: Tensor list of shape (90, 160, 256*w), (45, 80, 512*w), (22, 40, 512*w*r)
            masks: Tensor of shape (1, 1, 720, 1280)
            ms_features: First tensor of shape (1, 512, 45, 80)

        Returns:
            F_final: Tensor of shape (1, 512, 45, 80)
        """
        assert len(yolo_features) == 3
        for i in range(len(yolo_features)):
            yolo_features[i] = yolo_features[i].permute(1, 2, 0).unsqueeze(0)
        [yolo_f1, yolo_f2, yolo_f3] = yolo_features
        ms_feature = ms_features[0]
        
        # 1. 空间对齐 + 通道调整
        yolo_f1_down = F.avg_pool2d(yolo_f1, 2, 2)
        yolo_f1_down = self.conv_yolo1(yolo_f1_down)
        
        yolo_f2_aligned = self.conv_yolo2(yolo_f2)
        if self.w != 1:
            yolo_f2_aligned = F.interpolate(yolo_f2_aligned, scale_factor=self.w, mode='bilinear', align_corners=False)
        
        yolo_f3_up = F.interpolate(yolo_f3, size=(45, 80), mode='bilinear', align_corners=False)
        yolo_f3_up = self.conv_yolo3(yolo_f3_up)
        
        # 2. YOLO 多路特征融合
        yolo_concat = torch.cat([yolo_f1_down, yolo_f2_aligned, yolo_f3_up], dim=1)
        yolo_fused = self.fusion_conv(yolo_concat)
        
        # 3. 特征融合（Gating）
        F_concat = torch.cat([yolo_fused, ms_feature], dim=1)
        
        G = self.gate_conv(F_concat)
        G = torch.sigmoid(G)
        
        F_gate = G * yolo_fused + (1 - G) * ms_feature
        
        # 4. 最终 3x3 卷积精细化
        F_final = self.final_conv(F_gate)
        
        return F_final
