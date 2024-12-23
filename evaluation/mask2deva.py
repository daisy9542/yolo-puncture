"""
将图像分割模型推理的掩码图片使用 DEVA 处理
"""
from os import path, makedirs, listdir
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import json
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from pathlib import Path
from typing import List, Tuple

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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# Define a custom VideoReader to read mask images
class MaskVideoReader:
    def __init__(self, mask_dir: str, video_name: str):
        self.video_dir = path.join(mask_dir, video_name)
        self.frame_files = sorted([f for f in listdir(self.video_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.frame_files)
    
    def __getitem__(self, idx):
        frame_path = path.join(self.video_dir, self.frame_files[idx])
        # Load mask image
        mask_img = cv2.imread(frame_path)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        return mask_img, frame_path


# Function to convert mask image to binary mask and ObjectInfo
def load_mask(mask_img: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, List[ObjectInfo]]:
    """
    Converts a mask image to a binary mask tensor and creates ObjectInfo.
    Assumes mask_img is RGB with pixels either [0,0,0] or [255,0,0].
    """
    # Create binary mask: 1 where red channel is 255, else 0
    binary_mask = (mask_img[:, :, 0] == 255).astype(np.uint8)
    mask_tensor = torch.from_numpy(binary_mask).to(device)
    
    # Since there's only one instance, create a single ObjectInfo
    if binary_mask.any():
        segments_info = [ObjectInfo(id=1, score=1.0, category_id=0)]  # Adjust category_id as needed
    else:
        segments_info = []
    
    return mask_tensor, segments_info


def no_collate(x):
    return x


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
                  frame_path: str,
                  result_saver: ResultSaver,
                  ti: int,
                  *,
                  device: torch.device,
                  mask_dir: str,
                  video_name: str,
                  keyframe_selection: str = 'first'
                  ) -> None:
    """
    Processes a single frame using pre-segmented mask images.
    """
    # Load mask image
    mask_img = cv2.imread(frame_path)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    
    frame_path_obj = Path(frame_path)
    
    image_path_obj = Path('/home/puncture/datasets/needle-seg-full/images') / frame_path_obj.parent.name / (
                frame_path_obj.stem + '.jpg')
    image_path = str(image_path_obj)
    
    # Convert mask image to binary mask and segments_info
    mask, segments_info = load_mask(mask_img, device)
    
    cfg = deva.config
    
    h, w = mask.shape
    new_min_side = cfg['size']
    suppress_small_mask = cfg['suppress_small_objects']
    num_voting_frames = cfg['num_voting_frames']
    num_voting_frames += cfg['forward_clip_frames']  # Add forward frames
    need_resize = new_min_side > 0
    
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image = get_input_frame_for_deva(image_np, new_min_side)
    
    # Resize mask if needed
    if need_resize:
        scale = new_min_side / min(h, w)
        resized_mask = F.resize(mask.unsqueeze(0).float(), size=[int(h * scale), int(w * scale)]).squeeze(0)
    else:
        resized_mask = mask.float()
    
    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti,
                           {
                               'frame': [frame_name],
                               'shape': [h, w],
                           }
                           )
    
    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np
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
                                       image_np=image_np
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
    
    # For id2rgb
    np.random.seed(42)
    
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument("--parent_dir", type=str, required=True,
                        help="Parent directory containing 'pan_pred' and to save 'deva_pred'")
    add_custom_eval_args(parser)
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)
    args = parser.parse_args()
    
    # Extract video names from pan_pred directory
    pan_pred_dir = path.join(args.parent_dir, 'pan_pred')
    video_names = [d for d in listdir(pan_pred_dir) if path.isdir(path.join(pan_pred_dir, d))]
    
    # Prepare output directory
    deva_pred_dir = path.join(args.parent_dir, 'deva_pred')
    makedirs(deva_pred_dir, exist_ok=True)
    
    # Get DEVA model and configuration
    deva_model, cfg, _ = get_model_and_config(parser)
    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']
    cfg['enable_long_term_count_usage'] = False
    print('Configuration:', cfg)
    
    # Initialize ImageFeatureStore and DEVAInferenceCore
    image_feature_store = ImageFeatureStore(deva_model)
    deva = DEVAInferenceCore(deva_model, cfg, image_feature_store=image_feature_store)
    deva.next_voting_frame = args.num_voting_frames - 1
    deva.enabled_long_id()
    
    # Initialize ResultSaver for each video
    for video_name in video_names:
        print(f"Processing video: {video_name}")
        video_pan_pred_dir = path.join(pan_pred_dir, video_name)
        video_deva_pred_dir = path.join(deva_pred_dir, video_name)
        makedirs(video_deva_pred_dir, exist_ok=True)
        
        video_reader = MaskVideoReader(pan_pred_dir, video_name)
        loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
        result_saver = ResultSaver(video_deva_pred_dir, video_name, dataset='demo', object_manager=deva.object_manager)
        
        with torch.amp.autocast('cuda', enabled=args.amp):
            for ti, (mask, im_path) in enumerate(tqdm(loader, desc=f"Video {video_name}")):
                process_frame(deva, im_path, result_saver, ti,
                              device=device,
                              mask_dir=pan_pred_dir,
                              video_name=video_name,
                              keyframe_selection=args.keyframe_selection
                              )
            flush_buffer(deva, result_saver)
        result_saver.end()
        torch.cuda.empty_cache()
        
        # Save video-level JSON
        with open(path.join(video_deva_pred_dir, 'pred.json'), 'w') as f:
            json.dump(result_saver.video_json, f, indent=4)  # Prettier JSON
    
    print("Processing completed. Results are saved in 'deva_pred' directory.")
