import torch
import numpy as np
import imageio
from sam2.build_sam import build_sam2_video_predictor

from needle_seg.utils import get_config
from needle_seg.vis_prompt.video_top_mask import yolo_top_mask

CONFIG = get_config()

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
    video_num = 9
    yolo_model_path = f'{CONFIG.PATH.WEIGHTS_PATH}/seg/yolo11l-seg-finetune.pt'
    source = f'{CONFIG.PATH.DATASETS_PATH}/videos/video{video_num}.mp4'
    binary_masks = process_video_with_yolo_and_sam2(yolo_model_path, source, sam2_model_type='b+')
    # 将二值掩码列表转换为视频
    video_path = f'./video{video_num}_masks.mp4'
    rgb_masks = [np.stack([bi_mask.cpu().numpy()] * 3, axis=-1) * 255 for bi_mask in binary_masks]
    imageio.mimsave(video_path, rgb_masks, fps=30, codec='h264')
    print(f"Video saved to {video_path}")
