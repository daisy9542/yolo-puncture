import torch
import numpy as np
from yolo_seg.tasks.models.U2Net import U2NET, U2NETP

import sys
import os

# 添加yolo_seg模块到搜索目录
current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_seg_dir = os.path.join(current_dir, '../..')
sys.path.insert(0, yolo_seg_dir)


from yolo_seg import (
    numpy2tensor,
)

__all__ = [
    'load_unet',
    'unet_predict',
]


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def load_unet(model_name='u2netp', model_dir='', device='cuda'):
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.to(device)
    net.eval()

    return net




def unet_predict(model, image,  device='cuda'):
    input = numpy2tensor(image)
    if input.ndimension() == 3:
        # 如果是3D张量，则添加一个批次维度
        input = input.unsqueeze(0)
    inputs_test = input.to(device)
    d1,d2,d3,d4,d5,d6,d7= model(inputs_test)
    del d2,d3,d4,d5,d6,d7
    # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)


    # 转换为0-1 mask
    threshold = 0.5
    pred = pred.squeeze()  # Remove unnecessary dimensions
    predict_np = pred.cpu().data.numpy()  # Convert to numpy array
    # Convert the probability matrix to a binary mask
    # binary_mask = (predict_np > threshold).astype(np.uint8)  # Thresholding
    binary_mask = np.where(predict_np > threshold, 255, 0).astype(np.uint8)
    # print(binary_mask)
    return binary_mask