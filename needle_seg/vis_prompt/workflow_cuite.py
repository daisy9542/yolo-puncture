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

from needle_seg.vis_prompt.video_top_mask import yolo_top_mask
from needle_seg.utils import get_config
from cutie.inference.inference_core import InferenceCore
from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg

CONFIG = get_config()


def get_cutie_model() -> CUTIE:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    initialize(version_base='1.3.2', config_path="../../cutie/config", job_name="eval_config")
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


if __name__ == '__main__':
    video_num = 9
    yolo_model_path = f'{CONFIG.PATH.WEIGHTS_PATH}/seg/yolo11l-seg-finetune.pt'
    source = f'{CONFIG.PATH.DATASETS_PATH}/videos/video{video_num}.mp4'
    binary_masks = process_video_with_yolo_and_cutie(yolo_model_path, source)
    
    # 将二值掩码列表转换为视频
    video_path = f'./video{video_num}_masks.mp4'
    rgb_masks = [np.stack([bi_mask.cpu().numpy()] * 3, axis=-1) * 255 for bi_mask in binary_masks]
    imageio.mimsave(video_path, rgb_masks, fps=30, codec='h264')
    print(f"Video saved to {video_path}")
