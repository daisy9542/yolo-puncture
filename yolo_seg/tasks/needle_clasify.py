import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from timm.models import create_model, load_checkpoint

import models.van
import models.efficientnet

from utils import (get_config, crop_frame)

CONFIG = get_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 2
INPUT_IMG_SIZE = 380

__all__ = [
    'load_classify_net',
    'predict_images',
    'predict_and_find_start_inserted',
]


class ModifiedEfficientNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedEfficientNet, self).__init__()
        self.original_model = original_model
    
    def forward(self, x):
        return self.original_model(x)



# 加载保存的模型权重
def _load_net(model_name, num_classes=2, checkpoint=None, device=device):
    """
    model_name: 支持van和efficientnet 两种网络
    num_classes: 分类数
    checkpoint: 模型权重路径，,pt,.pth,.pth.tar文件路径
    """
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        in_chans=3,
        global_pool=None)

    # print(model)
    if checkpoint:
        load_checkpoint(model, checkpoint)

    model.to(device)
    model.eval()
    
    return model


def load_classify_net(checkpoint_name, device=device):
    return _load_net('efficientnet_b3', os.path.join(CONFIG.PATH.WEIGHTS_PATH, checkpoint_name), device=device)


def predict_images(model, images):
    """
      功能：预测类别
      model 模型
      iamges numpy.ndarray类型的图片数组
      return (indices, probabilities); indices是类别索引, probabilities是类别对应的概率列表
    """
    model.eval()
    probabilities = []
    indices = []
    
    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 
    transformed_images = [transform(Image.fromarray(image)) for image in images]
    
    batch = torch.stack(transformed_images).to(device)
    outputs = model(batch)
    probs = torch.softmax(outputs, dim=1)
    # Get the maximum probabilities and their corresponding indices
    max_probs, max_indices = torch.max(probs, dim=1)
    
    probabilities.extend(max_probs.detach().cpu().numpy())
    indices.extend(max_indices.detach().cpu().numpy())
    
    return indices, probabilities


def fix_class_prob(class_list, prob_list, class_index):
    """
      功能：修正分类-概率序列
    """
    n = len(class_list)
    # 向前遍历，从 class_index-1 到 0
    for i in range(class_index - 1, -1, -1):
        if class_list[i] != 0:
            # 向前搜索最近的0的概率
            found_prob = 0.6
            for j in range(i - 1, -1, -1):
                if class_list[j] == 0:
                    found_prob = prob_list[j]
                    break
            class_list[i] = 0
            prob_list[i] = found_prob
    
    # 向后遍历，从 class_index+1 到 n-1
    for i in range(class_index + 1, n):
        if class_list[i] != 1:
            # 向后搜索最近的1的概率
            found_prob = 0.6
            for j in range(i + 1, n):
                if class_list[j] == 1:
                    found_prob = prob_list[j]
                    break
            class_list[i] = 1
            prob_list[i] = found_prob
    
    return class_list, prob_list


def predict_and_find_start_inserted(model, frames=None, boxes_list=None, judge_wnd=20, batch_size=8):
    """
      功能：预测类别、概率和插入帧
      model
      frames 视频帧
      boxes_list 目标检测的框列表
      batch_size 分类预测的批量大小
      return 索引
    """
    if boxes_list is None:
        boxes_list = []
    if frames is None:
        frames = []
    if len(frames) != len(boxes_list):
        raise ValueError("The length of frames and boxes_list must be the same.")
    print("Start predict all frames ... ")
    roi_list = []
    previous_box = None
    # print(f"frames length: {len(frames)}")
    for i, xyxy in enumerate(boxes_list):
        frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        roi,_ = crop_frame(frame, xyxy, INPUT_IMG_SIZE, need_padding=True)
        roi_list.append(roi)
    
    # 预测
    class_list = []
    prob_list = []
    
    for i in range(0, len(roi_list), batch_size):
        batch_roi = roi_list[i:i + batch_size]
        classes, probs = predict_images(model, batch_roi)
        class_list.extend(classes)
        prob_list.extend(probs)
    
    ############# 找关键插入帧 START  #############
    required_count = 0.9 * judge_wnd
    
    # 阈值列表，从大到小排列
    thresholds = [0.9, 0.8, 0.7, 0.6]
    insert_frame_index = -1
    
    for i in range(len(prob_list) - judge_wnd + 1):
        wnd_probs = prob_list[i:i + judge_wnd]
        wnd_classes = class_list[i:i + judge_wnd]
        
        # 计算当前窗口内 `class=1` 的帧数
        count = sum(1 for j in range(judge_wnd) if wnd_classes[j] == 1)
        
        if count >= required_count:
            # 遍历各个阈值
            for threshold in thresholds:
                # 寻找连续5帧 `class=1` 且概率 > 阈值
                for k in range(judge_wnd - 4):
                    if all(wnd_classes[k + l] == 1 and wnd_probs[k + l] > threshold for l in range(5)):
                        insert_frame_index = i + k
                        break
                if insert_frame_index != -1:
                    break
            if insert_frame_index != -1:
                break
    if insert_frame_index == -1:
        insert_frame_index = 0
    ############# 找关键插入帧 END  #############
    
    # 分类序列修正 
    fix_class_prob(class_list, prob_list, insert_frame_index)
    
    return class_list, prob_list, insert_frame_index
