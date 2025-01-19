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
from needle_seg.utils import get_config
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
    parser.add_argument('--video_num', type=int, help='The number of the video to process')
    args = parser.parse_args()
    yolo_model_path = f'{CONFIG.PATH.WEIGHTS_PATH}/seg/yolo11l-seg-finetune.pt'
    
    mode = args.mode
    video_num = args.video_num
    source = f'{CONFIG.PATH.DATASETS_PATH}/videos/video{video_num}.mp4'
    
    # 处理视频并生成二值掩码
    if mode == 'cutie':
        binary_masks = process_video_with_yolo_and_cutie(yolo_model_path, source, conf=0.25, top_percent=0.05, device='0')
    elif mode == 'sam2':
        binary_masks = process_video_with_yolo_and_sam2(yolo_model_path, source, sam2_model_type='b+')
    
    # 保存二值掩码视频
    mask_video_path = f'./video{video_num}_masks.mp4'
    rgb_masks = [np.stack([bi_mask.cpu().numpy()] * 3, axis=-1) * 255 for bi_mask in binary_masks]
    imageio.mimsave(mask_video_path, rgb_masks, fps=30, codec='h264')
    print(f"二值掩码视频已保存到 {mask_video_path}")
    
    # 生成叠加掩码的视频
    overlay_video_path = f'./video{video_num}_overlay.mp4'
    
    # 使用上下文管理器确保读取器和写入器正确关闭
    with imageio.get_reader(source) as reader, imageio.get_writer(overlay_video_path, fps=30, codec='h264') as writer:
        alpha = 0.5  # 透明度因子，0.0 完全透明，1.0 完全不透明
        color = [255, 0, 0]  # 掩码颜色，这里为红色
        
        for frame_idx, (frame, mask) in enumerate(zip(reader, binary_masks)):
            frame = frame.astype(np.uint8)
            mask_np = mask.cpu().numpy().astype(np.uint8)
            
            # 确保掩码是二值的（0 或 1）
            mask_np = (mask_np > 0).astype(np.uint8)
            
            # 创建彩色掩码
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 0] = color[0]  # 红色通道
            colored_mask[:, :, 1] = color[1]  # 绿色通道
            colored_mask[:, :, 2] = color[2]  # 蓝色通道
            
            # 扩展掩码到3个通道
            mask_3c = np.stack([mask_np]*3, axis=-1)
            
            # 叠加掩码到原始帧
            overlaid_frame = frame.copy()
            overlaid_frame[mask_3c > 0] = (
                frame[mask_3c > 0] * (1 - alpha) + colored_mask[mask_3c > 0] * alpha
            ).astype(np.uint8)
            
            writer.append_data(overlaid_frame)
            
            if (frame_idx + 1) % 100 == 0:
                print(f"已处理 {frame_idx + 1} 帧")
    
    print(f"叠加掩码视频已保存到 {overlay_video_path}")
