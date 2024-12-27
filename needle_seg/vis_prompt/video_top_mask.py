import numpy as np
from ultralytics import YOLO


def yolo_top_mask(model_path: str, source: str, **kwargs):
    """
    使用 YOLO 模型进行视频分割，筛选出置信度最高的掩码，并根据置信度筛选前 top_percent% 的帧。

    Args:
        model_path: YOLO 模型路径。
        source: 视频的路径。
        conf: 置信度阈值，低于此值的目标将被忽略。默认值是 0.25。
        top_percent: 用于选择置信度前多少百分比帧的参数。默认是 10%（0.1）。
        device: 运行设备。可以为 'cpu' 或 CUDA ID。默认是 '0'。

    Return:
        top_percent_frames: 置信度前 top_percent% 的帧及其对应的掩码。
    """
    conf = kwargs.get('conf', 0.25)
    top_percent = kwargs.get('top_percent', 0.1)
    device = kwargs.get('device', '0')
    
    # 使用YOLO进行预测
    if device == 'cpu':
        pass
    elif device.isnumeric():
        device = f'cuda:{device}'
    else:
        raise ValueError(f"Invalid device: {device}")
    if not source.endswith('.mp4'):
        raise ValueError(f"Invalid source: {source}")
    model = YOLO(model_path)
    results = model.predict(source, device=device, conf=conf, retina_masks=True, stream=True)
    
    frame_info = []
    
    for i, result in enumerate(results):
        # 获取每帧的掩码和置信度
        confidences = result.boxes.conf.cpu().numpy()  # 置信度数组
        frame_id = i  # 当前帧的 ID
        height, width = result.orig_shape
        
        # 选择每帧置信度最高的掩码
        if len(confidences) > 0:
            masks = result.masks.data.cpu().numpy()  # 形状为 (n * H * W)
            max_confidence_idx = np.argmax(confidences)  # 获取最大置信度的索引
            max_confidence_mask = masks[max_confidence_idx]  # 获取对应的掩码
            max_confidence = confidences[max_confidence_idx]  # 获取最大置信度
            
            frame_info.append({
                'frame_id': frame_id,
                'mask': max_confidence_mask,
                'confidence': max_confidence
            })
        else:
            frame_info.append({
                'frame_id': frame_id,
                'mask': np.zeros((height, width), dtype=np.uint8),
                'confidence': 0
            })
    
    # 排序：根据置信度选择 top_percent 的帧
    frame_confidences_sorted = sorted(frame_info, key=lambda x: x['confidence'], reverse=True)
    top_n_frames = int(len(frame_confidences_sorted) * top_percent)
    top_percent_frames = frame_confidences_sorted[:top_n_frames]
    
    return top_percent_frames
