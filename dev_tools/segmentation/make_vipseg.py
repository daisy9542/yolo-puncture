import os
import re
import cv2
import json
import numpy as np
from tqdm import tqdm
import argparse
import concurrent.futures
from needle_seg.utils import polygon_encoding


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]


def process_frame(frame_idx, video_frame, mask_frame, video_name, images_dir, panomasks_dir):
    """处理每一帧的图像和掩码，并更新 global_json。"""
    h, w = video_frame.shape[:2]
    frame_name = f"{frame_idx:04d}"
    
    # 保存图像和掩码
    cv2.imwrite(os.path.join(images_dir, video_name, f"{frame_name}.jpg"), video_frame)
    cv2.imwrite(os.path.join(panomasks_dir, video_name, f"{frame_name}.png"), mask_frame)
    
    # 初始化 RGB panoptic mask
    panoptic_mask_RGB = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(mask_frame)
    segments_info = []
    
    for obj_id in unique_ids:
        if obj_id == 0:
            continue
        
        assigned_id = 1  # 固定实例 ID 为 1
        color = (0, 0, 255)
        panoptic_mask_RGB[mask_frame == obj_id] = color
        
        bbox = cv2.boundingRect((mask_frame == obj_id).astype(np.uint8))
        
        # 使用 polygon_encoding 来生成归一化的多边形坐标
        binary_segment = (mask_frame == obj_id).astype(np.uint8)
        segmentation = polygon_encoding(binary_segment, normalize=False)
        
        segments_info.append({
            "id": assigned_id,
            "category_id": 1,
            "area": int(np.sum(mask_frame == obj_id)),
            "bbox": list(bbox),
            "iscrowd": 0,
            "segmentation": [segmentation]
        })


def create_vipseg_dataset(video_dir, mask_dir, output_dir):
    images_dir = os.path.join(output_dir, "images")
    panomasks_dir = os.path.join(output_dir, "panomasks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(panomasks_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    video_files = sorted(video_files, key=natural_sort_key)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_dir, video_file)
            mask_video_path = os.path.join(mask_dir, video_file)
            
            if not os.path.exists(mask_video_path):
                print(f"Mask video for '{video_name}' does not exist. Skipping.")
                continue
            
            # 创建视频文件输出路径
            os.makedirs(os.path.join(images_dir, video_name), exist_ok=True)
            os.makedirs(os.path.join(panomasks_dir, video_name), exist_ok=True)
            
            cap_video = cv2.VideoCapture(video_path)
            cap_mask = cv2.VideoCapture(mask_video_path)
            
            frame_idx = 0
            while True:
                ret_video, video_frame = cap_video.read()
                ret_mask, mask_frame = cap_mask.read()
                
                if not ret_video or not ret_mask:
                    break
                
                if video_frame is None or mask_frame is None:
                    frame_idx += 1
                    continue
                
                if len(mask_frame.shape) == 3:
                    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                
                mask_frame[mask_frame > 100] = 255
                mask_frame[mask_frame <= 100] = 0
                
                # 将任务提交给线程池
                futures.append(executor.submit(process_frame, frame_idx, video_frame, mask_frame, video_name,
                                               images_dir, panomasks_dir))
                
                frame_idx += 1
            
            cap_video.release()
            cap_mask.release()
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            global_json = future.result()
    
    # 保存生成的 JSON 文件
    json_path = os.path.join(output_dir, "panoptic_gt_VIPSeg.json")
    with open(json_path, "w") as f:
        json.dump(global_json, f, indent=4)
    
    print(f"VIPSeg dataset saved to '{output_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a yolo-seg dataset to VIPSeg format')
    parser.add_argument('videos', type=str, help='Path to the source video directory')
    parser.add_argument('masks', type=str, help='Path to the source mask video directory')
    parser.add_argument('target', type=str, help='Path to the target VIPSeg dataset directory')
    args = parser.parse_args()
    create_vipseg_dataset(args.videos, args.masks, args.target)
