import math
import torch
from typing import Optional, Union, Tuple


def get_similarity(mk: torch.Tensor,
                   ms: torch.Tensor,
                   qk: torch.Tensor,
                   qe: torch.Tensor,
                   add_batch_dim=False
                   ) -> torch.Tensor:
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    if add_batch_dim:
        mk, ms = mk.unsqueeze(0), ms.unsqueeze(0)
        qk, qe = qk.unsqueeze(0), qe.unsqueeze(0)
    
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None
    
    if qe is not None:
        # See XMem's appendix for derivation
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq + two_ab - b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq + two_ab)
    
    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)  # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)  # B*N*HW
    
    return similarity


def do_softmax(
        similarity: torch.Tensor,
        top_k: Optional[int] = None,
        inplace: bool = False,
        return_usage: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)
        
        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)  # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)  # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None
    
    if return_usage:
        return affinity, affinity.sum(dim=2)
    
    return affinity


def get_affinity(mk: torch.Tensor, ms: torch.Tensor, qk: torch.Tensor,
                 qe: torch.Tensor
                 ) -> torch.Tensor:
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity


def readout(affinity: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    B, CV, T, H, W = mv.shape
    
    mo = mv.view(B, CV, T * H * W)
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)
    
    return mem


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, size, scale_factors=[1, 0.5, 0.25], r=1):
        super().__init__()
        self.size = size
        
        # 卷积层用于融合不同分辨率的特征
        self.conv_wxh = nn.Conv2d(in_channels + 1, mid_channels, size, padding=(size // 2))  # Convw×h
        self.conv_1x1 = nn.Conv2d(mid_channels, out_channels, 1)  # Conv1×1
        
        # 用于调整第三特征的通道数（如果有系数r）
        self.conv_third_feature = nn.Conv2d(in_channels * r, in_channels, 1) if r > 1 else nn.Identity()
    
    def forward(self, features, prev_frame_mask):
        # features 是一个包含 [small, medium, large] 特征的列表
        # prev_frame_mask 是上一帧的掩码
        small_feature, medium_feature, large_feature = features
        
        # 对齐特征到相同分辨率（取小特征的分辨率为基准）
        small_feature_resized = small_feature
        medium_feature_resized = F.interpolate(medium_feature, small_feature.shape[-2:], mode="bilinear")
        large_feature_resized = F.interpolate(large_feature, small_feature.shape[-2:], mode="bilinear")
        
        # 处理第三特征的通道数（考虑系数 r）
        large_feature_resized = self.conv_third_feature(large_feature_resized)
        
        # 拼接多尺度特征
        fused_feature = torch.cat([small_feature_resized, medium_feature_resized, large_feature_resized], dim=1)
        
        # 处理上一帧的掩码
        prev_frame_mask = F.interpolate(prev_frame_mask, fused_feature.shape[-2:], mode="bilinear")
        concatenated_features = torch.cat((prev_frame_mask, fused_feature), dim=1)
        
        # 卷积提取注意力特征
        local_attention_feature = self.conv_wxh(concatenated_features)
        local_attention_feature = self.conv_1x1(local_attention_feature)
        
        # Softmax 归一化
        alpha = F.softmax(local_attention_feature, dim=1)
        
        # 加权融合特征与上一帧掩码
        fused_output = alpha * prev_frame_mask
        
        return fused_output
