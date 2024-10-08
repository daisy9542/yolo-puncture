from os import path
from argparse import ArgumentParser
import re
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer, get_input_frame_for_deva
from deva.inference.frame_utils import FrameInfo
from deva.inference.object_info import ObjectInfo
from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args
from deva.utils.tensor_utils import pad_divide_by, unpad

from ultralytics import YOLO

from tqdm import tqdm
import json
import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import cv2


def sort_key(filename):
    # 使用正则表达式提取前面和后面的数字
    match = re.match(r"(\d+)[^\d]+(\d+)\.jpg", filename)
    if match:
        # 将提取到的数字部分转换为整数用于排序
        return int(match.group(1)), int(match.group(2))
    return 0, 0  # 如果没有匹配成功返回一个默认值


def no_collate(x):
    return x


class VideoReader(Dataset):
    """
    This class is used to read a video, one frame at a time
    This simple version:
    1. Does not load the mask/json
    2. Does not normalize the input
    3. Does not resize
    """
    
    def __init__(
            self,
            image_dir,
    ):
        """
        image_dir - points to a directory of jpg images
        """
        self.image_dir = image_dir
        self.frames = sorted(os.listdir(self.image_dir), key=sort_key)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')
        img = np.array(img)
        
        return img, im_path
    
    def __len__(self):
        return len(self.frames)


def generate_output_mask(masks_xy: torch.Tensor, ids: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    """
    将 YOLO 输出的掩码和 ID 列表转换为 output_mask 格式
    :param masks_xy: YOLO 输出的掩码，类型为 List[np.ndarray]，每个元素是形状为 (N, 2) 的数组
    :param ids: YOLO 输出的 id 列表，形状为 (num,)
    :param original_size: 原始图像的大小，形状为 h * w
    :param device: 创建张量所在的设备
    :return: 形状为 h * w 的 output_mask
    """
    h, w = original_size
    output_mask = torch.zeros((h, w), dtype=torch.int64, device="cuda")
    if ids is None or masks_xy is None:
        return output_mask
    
    # 遍历每个掩码并将其应用到output_mask上
    for mask, id_ in zip(masks_xy, ids):
        # mask 是形状为 (N, 2) 的数组，包含掩码的像素坐标
        mask = np.array(mask)  # 确保mask是numpy数组
        x_coords, y_coords = mask[:, 0], mask[:, 1]  # 获取x和y坐标
        
        # 将掩码位置标记为相应的ID
        for x, y in zip(x_coords, y_coords):
            if 0 <= x < w and 0 <= y < h:  # 确保坐标在范围内
                output_mask[int(y), int(x)] = id_  # 注意这里的y, x顺序
    
    return output_mask


def estimate_forward_mask(deva: DEVAInferenceCore, image: torch.Tensor):
    image, pad = pad_divide_by(image, 16)
    image = image.unsqueeze(0)  # add the batch dimension
    
    ms_features = deva.image_feature_store.get_ms_features(deva.curr_ti + 1, image)
    key, _, selection = deva.image_feature_store.get_key(deva.curr_ti + 1, image)
    prob = deva._segment(key, selection, ms_features)
    forward_mask = torch.argmax(prob, dim=0)
    forward_mask = unpad(forward_mask, pad)
    return forward_mask


@torch.inference_mode()
def process_frame(deva: DEVAInferenceCore,
                  yolo_model: YOLO,
                  frame_path: str,
                  result_saver: ResultSaver,
                  ti: int,
                  image_np: np.ndarray = None) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config
    
    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    suppress_small_mask = cfg['suppress_small_objects']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)
    
    result = yolo_model.track(image_np)[0]
    masks_xy = result.masks.xy if result.masks else None
    ids = result.boxes.id
    orig_shape = result.orig_shape
    output_mask = generate_output_mask(masks_xy, ids, orig_shape)
    segments_info = [
        ObjectInfo(
            id=i + 1,
            category_id=result.boxes.cls[i],
            score=result.boxes.conf[i],
        )
        for i in range(result.boxes.cls.numel())
    ]
    
    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, output_mask, segments_info, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })
    
    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            # getting a forward mask
            if deva.memory.engaged:
                forward_mask = estimate_forward_mask(deva, image)
            else:
                forward_mask = None
            
            # mask, segments_info = make_segmentation(cfg, image_np, forward_mask, sam_model,
            #                                         new_min_side, suppress_small_mask)
            # frame_info.mask = mask
            # frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)
            
            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np
                
                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image,
                                                  mask,
                                                  new_segments_info,
                                                  incremental=True)
                deva.next_voting_frame += cfg['detection_every']
                
                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np)
                
                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np)
                
                deva.clear_buffer()
        else:
            # standard propagation
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np)
    
    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            if deva.memory.engaged:
                forward_mask = estimate_forward_mask(deva, image)
            else:
                forward_mask = None
            
            # mask, segments_info = make_segmentation(cfg, image_np, forward_mask, sam_model,
            #                                         new_min_side, suppress_small_mask)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, output_mask, segments_info, incremental=True)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np)


if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)
    
    # for id2rgb
    np.random.seed(42)
    """
    Arguments loading
    """
    parser = ArgumentParser()
    
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)
    deva_model, cfg, args = get_model_and_config(parser)
    yolo_model = YOLO("seg/yolo11n-seg-finetune.pt")
    # sam_model = get_sam_model(cfg, 'cuda')
    """
    Temporal setting
    """
    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']
    
    # get data
    video_reader = VideoReader(cfg['img_path'])
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
    out_path = cfg['output']
    
    # Start eval
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    cfg['enable_long_term_count_usage'] = (
            cfg['enable_long_term']
            and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
                 cfg['num_prototypes']) >= cfg['max_long_term_elements'])
    
    print('Configuration:', cfg)
    
    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = args.num_voting_frames - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, "deva-result.mp4", dataset='demo', object_manager=deva.object_manager)
    
    with torch.cuda.amp.autocast(enabled=args.amp):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            process_frame(deva, yolo_model, im_path, result_saver, ti, image_np=frame)
        flush_buffer(deva, result_saver)
    result_saver.end()
    
    # save this as a video-level json
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json
