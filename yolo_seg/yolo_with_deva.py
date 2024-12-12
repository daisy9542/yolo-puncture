from os import path
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import json
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from typing import Dict, List, Tuple

from ultralytics import YOLO

from deva.inference.image_feature_store import ImageFeatureStore
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.custom_eval_args import add_custom_eval_args
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer, get_input_frame_for_deva
from deva.inference.frame_utils import FrameInfo
from deva.inference.object_info import ObjectInfo
from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args
from deva.utils.tensor_utils import pad_divide_by, unpad

from yolo_seg.utils.video_reader import VideoReader
from yolo_seg.utils.embedding_extractor import EmbeddingExtractor

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def no_collate(x):
    return x


def auto_segment(config: Dict, image: np.ndarray, yolo_model,
                 min_side: int, suppress_small_mask: bool,
                 *,
                 ti: int = None,
                 extractor: EmbeddingExtractor = None,
                 image_feature_store: ImageFeatureStore = None,
                 ) -> Tuple[torch.Tensor, List[ObjectInfo]]:
    """
    使用 YOLO 模型对图像进行实例分割，生成索引掩码和分割信息列表。
    """
    device = next(yolo_model.model.parameters()).device
    
    # 调整图像尺寸
    h, w = image.shape[:2]
    if min_side > 0:
        scale = min_side / min(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    
    # 模型推理
    results = yolo_model.predict(image, retina_masks=True)
    detections = results[0]
    if extractor is not None:
        extractor.attach_to_results(detections)
    
    output_mask = torch.zeros((h, w), dtype=torch.int64, device=device)
    segments_info = []
    curr_id = 1
    
    boxes = detections.boxes  # 边界框
    masks = detections.masks  # 分割掩码
    embeddings = detections.embeddings  # YOLO 三个尺度的特征
    if image_feature_store is not None:
        image_feature_store.push_yolo_features(ti, embeddings.values())
        
    if len(boxes.cls) > 0:
        # 若检测到多个物体，取置信度最大的
        best_conf_idx = np.argmax(boxes.conf.cpu())
        best_box = boxes.xyxy[best_conf_idx]
        best_mask = masks.data[best_conf_idx]
        
        # 使用最大置信度的掩码
        if isinstance(best_mask, np.ndarray):
            mask = torch.from_numpy(best_mask).to(device)
        else:
            mask = best_mask.to(device)
        
        # 如果掩码尺寸与输出掩码尺寸不一致，进行调整
        if mask.shape != (h, w):
            mask = F.resize(mask.unsqueeze(0), size=[h, w])[0]
        
        # 将掩码中为 True 的像素赋值为当前 ID
        output_mask[mask > 0.5] = curr_id
        
        # 获取置信度和类别信息
        score = boxes.conf[best_conf_idx].item()
        cls = boxes.cls[best_conf_idx].item()
        
        segments_info.append(ObjectInfo(id=curr_id, score=score, category_id=int(cls)))
        # curr_id += 1
    
    return output_mask, segments_info


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
                  *,
                  image_np: np.ndarray = None,
                  extractor: EmbeddingExtractor = None,
                  image_feature_store: ImageFeatureStore = None,
                  keyframe_selection: str = 'first'
                  ) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config
    
    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    suppress_small_mask = cfg['suppress_small_objects']
    num_voting_frames = cfg['num_voting_frames']
    num_voting_frames += cfg['forward_clip_frames']  # 加上前向帧内共识帧数
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)
    
    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti,
                           {
                               'frame': [frame_name],
                               'shape': [h, w],
                           }
                           )
    if torch.cuda.is_available():
        yolo_model.model.to(device)
    
    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            # getting a forward mask
            # if deva.memory.engaged:
            #     forward_mask = estimate_forward_mask(deva, image)
            # else:
            #     forward_mask = None
            
            mask, segments_info = auto_segment(cfg, image_np, yolo_model,
                                               new_min_side, suppress_small_mask,
                                               ti=ti,
                                               extractor=extractor,
                                               image_feature_store=image_feature_store,
                                               )
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)
            
            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np
                
                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection=keyframe_selection
                )
                prob = deva.incorporate_detection(this_image,
                                                  mask,
                                                  new_segments_info,
                                                  incremental=True
                                                  )
                deva.next_voting_frame += cfg['detection_every']
                
                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np
                                       )
                
                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np
                                           )
                
                deva.clear_buffer()
        else:
            # standard propagation
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np
                                   )
    
    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            # if deva.memory.engaged:
            #     forward_mask = estimate_forward_mask(deva, image)
            # else:
            #     forward_mask = None
            
            mask, segments_info = auto_segment(cfg, image_np, yolo_model,
                                               new_min_side, suppress_small_mask,
                                               ti=ti,
                                               extractor=extractor,
                                               image_feature_store=image_feature_store,
                                               )
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info, incremental=True)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np
                               )


if __name__ == '__main__':
    torch.autograd.set_grad_enabled(False)
    
    # for id2rgb
    np.random.seed(42)
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True, help="Save folder of the video")
    add_custom_eval_args(parser)
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)
    deva_model, cfg, args = get_model_and_config(parser)
    yolo_model = YOLO("seg/yolo11n-seg-finetune.pt")
    # sam_model = get_sam_model(cfg, 'cuda')
    extractor = EmbeddingExtractor(yolo_model)
    keyframe_selection = args.keyframe_selection
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
    
    image_feature_store = ImageFeatureStore(deva_model)
    deva = DEVAInferenceCore(deva_model, cfg, image_feature_store=image_feature_store)
    deva.next_voting_frame = args.num_voting_frames - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, cfg["video_name"], dataset='demo', object_manager=deva.object_manager)
    
    with torch.cuda.amp.autocast(enabled=args.amp):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            process_frame(deva, yolo_model, im_path, result_saver, ti,
                          image_np=frame, extractor=extractor,
                          image_feature_store=image_feature_store,
                          keyframe_selection=keyframe_selection
                          )
        flush_buffer(deva, result_saver)
    result_saver.end()
    
    # save this as a video-level json
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json
