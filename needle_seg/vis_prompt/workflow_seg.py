import os
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import imageio
import cv2
from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
import argparse
from sam2.build_sam import build_sam2_video_predictor

from needle_seg.vis_prompt.video_top_mask import yolo_top_mask
from needle_seg.utils import get_config, parse_video_range
from cutie.inference.inference_core import InferenceCore
from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg

CONFIG = get_config()


def get_cutie_model() -> CUTIE:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    initialize(version_base='1.3.2', config_path="../../../Cutie/cutie/config", job_name="eval_config")
    cfg = compose(config_name="eval_config")
    weight_dir = f'{CONFIG.PATH.WEIGHTS_PATH}/cutie'
    with open_dict(cfg):
        cfg['weights'] = os.path.join(weight_dir, 'cutie-base-mega.pth')
    get_dataset_cfg(cfg)
    
    # Load the network weights
    cutie = CUTIE(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights)
    cutie.load_weights(model_weights)
    
    return cutie


@torch.inference_mode()
@torch.amp.autocast('cuda')
def process_video_with_yolo_and_cutie(yolo_model_path, source, **kwargs):
    """
    使用 YOLO 模型提取视频中部分分割掩码，并使用 Cutie 模型对每一帧进行处理。

    Args:
        yolo_model_path (str): YOLO 模型的路径，用于提取视频帧中的目标掩码。
        source (str): 视频文件路径，供 YOLO 模型进行处理和 CUTIE 模型进一步推理。
        **kwargs: 可选的 YOLO 参数，包括 conf、top_percent 和 device。
            - conf (float): YOLO 置信度阈值，默认 0.25。
            - top_percent (float): 筛选置信度前 top_percent% 的帧，默认 0.05。
            - device (str): 运行设备，默认 '0'（即 GPU 0）。

    Return:
        mask_list (list): 二值掩码列表，每一帧的处理结果，np.uint8 类型。
    """
    
    top_percent_frames = yolo_top_mask(yolo_model_path, source, **kwargs)
    cutie = get_cutie_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    processor.max_internal_size = -1
    cap = cv2.VideoCapture(source)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first_mask = top_percent_frames[0]['mask']
    first_mask_id = top_percent_frames[0]['frame_id']
    seq = list(range(frame_count))
    seq[:first_mask_id + 1] = reversed(seq[:first_mask_id + 1])
    objects = np.unique(first_mask)
    objects = objects[objects != 0].tolist()
    objects = list(map(int, objects))
    mask_list = []
    for ti in tqdm(seq, desc='Processing frames by cutie', unit='frame'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, ti)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {ti}")
            continue
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = to_tensor(image_pil).cuda().float()
        mask_tensor = None
        if any(frame_info['frame_id'] == ti for frame_info in top_percent_frames):
            mask = next(frame_info['mask'] for frame_info in top_percent_frames if frame_info['frame_id'] == ti)
            mask_tensor = torch.from_numpy(mask).cuda()
        if mask_tensor is not None:
            output_prob = processor.step(image, mask_tensor, objects=objects)
        else:
            output_prob = processor.step(image)
        binary_mask = processor.output_prob_to_mask(output_prob)
        binary_mask = (binary_mask > 0).to(torch.uint8)
        mask_list.append(binary_mask)
    mask_list[:first_mask_id + 1] = reversed(mask_list[:first_mask_id + 1])
    cap.release()
    return mask_list


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

MODEL_TYPE_MAP = {
    't': 'tiny',
    's': 'small',
    'b+': 'base_plus',
    'l': 'large',
}


# sam2_hiera_tiny.pt sam2_hiera_small.pt sam2_hiera_base_plus.pt sam2_hiera_large.pt

@torch.inference_mode()
@torch.amp.autocast('cuda')
def process_video_with_yolo_and_sam2(yolo_model_path, source, sam2_model_type='b+', **kwargs):
    assert sam2_model_type in MODEL_TYPE_MAP.keys(), 'Unsupported sam2 model type, please choose one in t, s, b+ and l'
    top_percent_frames = yolo_top_mask(yolo_model_path, source, **kwargs)
    sam2_checkpoint = f'{CONFIG.PATH.WEIGHTS_PATH}/sam2/sam2.1_hiera_{MODEL_TYPE_MAP[sam2_model_type]}.pt'
    model_cfg = f'configs/sam2.1/sam2.1_hiera_{sam2_model_type}.yaml'
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
    state = predictor.init_state(video_path=source)
    
    for frame_info in top_percent_frames:
        predictor.add_new_mask(state, frame_info['frame_id'], 1, frame_info['mask'])
    
    video_segments = []
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        assert len(out_obj_ids) == 1
        binary_mask = (out_mask_logits[0, 0] > 0.0).to(torch.uint8)
        video_segments.append(binary_mask)
    
    return video_segments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process puncture video')
    parser.add_argument('mode', choices=['cutie', 'sam2'], help='The mode to process the video')
    parser.add_argument('--video_num', type=parse_video_range, help='The number of the video to process')
    args = parser.parse_args()
    yolo_model_path = f'{CONFIG.PATH.WEIGHTS_PATH}/seg/yolo11l-seg-finetune.pt'
    
    mode = args.mode
    for video_num in args.video_num:
        source = f'{CONFIG.PATH.DATASETS_PATH}/full-videos/video{video_num}.mp4'
        print(f"Processing {source}")
        
        # 处理视频并生成二值掩码
        if mode == 'cutie':
            binary_masks = process_video_with_yolo_and_cutie(
                yolo_model_path, source, conf=0.25, top_percent=0.05, device='0'
            )
        elif mode == 'sam2':
            binary_masks = process_video_with_yolo_and_sam2(
                yolo_model_path, source, sam2_model_type='b+'
            )
        
        os.makedirs(f'./workflow/{mode}', exist_ok=True)
        mask_video_path = f'./workflow/{mode}/video{video_num}_masks.mp4'
        
        # 使用 OpenCV 获取原视频的帧率
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者尝试 'X264'
        
        # 将二值掩码转换为 RGB 掩码
        rgb_masks = [np.stack([bi_mask.cpu().numpy()] * 3, axis=-1) * 255 for bi_mask in binary_masks]
        rgb_masks = [mask.astype(np.uint8) for mask in rgb_masks]
        
        if len(rgb_masks) == 0:
            raise ValueError("没有生成任何掩码！")
        
        video_writer = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height))
        
        # 写入每一帧到视频
        for frame in rgb_masks:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"二值掩码视频已保存到 {mask_video_path}")
        
        # 生成叠加掩码的视频
        overlay_video_path = f'./workflow/{mode}/video{video_num}_overlay.mp4'
        
        video_writer = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))
        
        alpha = 0.5  # 透明度因子
        color = [255, 0, 0]  # 红色
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx >= len(binary_masks):
                # 掩码已经用完，可以 break，也可以按需求处理
                break
            
            # 取对应帧的掩码（请确保 binary_masks 的长度与视频帧数匹配）
            mask = binary_masks[frame_idx]
            
            # 转为 uint8 类型
            frame = frame.astype(np.uint8)
            # 如果 frame 为灰度图，则转换为三通道图像
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # 获取 mask 的 numpy 数组，并二值化
            mask_np = mask.cpu().numpy().astype(np.uint8)
            # 如果 mask 是三通道，取第一通道
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            mask_np = (mask_np > 0).astype(np.uint8)
            
            # 保证 mask 尺寸与 frame 相同
            if mask_np.shape != frame.shape[:2]:
                mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 创建与 frame 大小相同的彩色掩码
            colored_mask = np.full(frame.shape, color, dtype=np.uint8)
            
            # 利用 cv2.addWeighted 计算整个帧的混合效果
            blended = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)
            
            # 将 mask 扩展到 (H, W, 1) 以便广播
            mask_expanded = mask_np[..., None].astype(bool)
            
            # 根据 mask 条件选取 blended 或原始 frame 像素
            overlaid_frame = np.where(mask_expanded, blended, frame)
            overlaid_frame = np.ascontiguousarray(overlaid_frame, dtype=np.uint8)
            
            # 写入视频帧
            video_writer.write(overlaid_frame)
            frame_idx += 1
        
        cap.release()
        video_writer.release()
        
        print(f"叠加掩码视频已保存到 {overlay_video_path}")
