import os
import re
import cv2
import numpy as np
import argparse
import pickle
from functools import cmp_to_key
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch
from utils.config import get_config
from utils.segment_everything import segment
from utils.mask_tools import filter_masks
from dev_tools.toolbox import KEY_FRAME, FRAME_OFFSET, polygon_encoding

CONFIG = get_config()

pattern = re.compile(r'.*?(\d+)frame_(\d+)\.(jpg|jpeg|png)')


def compare_filenames(f1, f2):
    match1 = pattern.match(f1)
    match2 = pattern.match(f2)
    
    if not match1 or not match2:
        raise ValueError(f"Filename does not match the expected pattern: {f1}, {f2}")
    
    video1, frame1 = map(int, match1.groups()[:2])
    video2, frame2 = map(int, match2.groups()[:2])
    
    if video1 != video2:
        return video1 - video2
    return frame1 - frame2


def process_video(video_num, input_dir, output_dir, device_id, topn):
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if pattern.match(file):
                match = pattern.match(file)
                cur_video_num, frame_num = map(int, match.groups()[:2])
                start_frame, end_frame = KEY_FRAME[video_num]
                start_frame -= FRAME_OFFSET
                end_frame += FRAME_OFFSET
                if cur_video_num == video_num and start_frame <= frame_num <= end_frame:
                    files_to_process.append(os.path.join(root, file))
    
    output_file_path = os.path.join(output_dir, f"video{video_num}.pkl")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    video_masks = {}
    sorted_files = sorted(files_to_process, key=cmp_to_key(compare_filenames))
    
    torch.cuda.set_device(device_id)
    
    for input_file_path in tqdm(sorted_files, desc=f"Processing video {video_num}"):
        image = cv2.imread(input_file_path)
        
        masks = segment(image)
        masks = filter_masks(masks, topn)
        
        polygon_masks = []
        filename = os.path.basename(input_file_path)
        for mask in masks:
            polygon_masks.append({
                "segmentation": polygon_encoding(mask["segmentation"]),
                "bbox": mask["bbox"]
            })
        video_masks[filename] = polygon_masks
    pickle.dump(video_masks, open(output_file_path, 'wb'))


def parse_numbers(s):
    if s.isdigit():
        return [int(s)]
    elif '-' in s:
        start, end = map(int, s.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(item) for item in s.split(',')]


def run(video_nums, input_dir, output_dir, num_videos_per_gpu, topn):
    num_gpus = torch.cuda.device_count()
    max_workers = num_gpus * num_videos_per_gpu
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, video_num in enumerate(video_nums):
            device_id = (i // num_videos_per_gpu) % num_gpus
            futures.append(executor.submit(process_video, video_num, input_dir, output_dir, device_id, topn))
        
        for future in tqdm(futures):
            future.result()  # 等待所有任务完成


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="segment annotations")
    parser.add_argument('--video_num', type=str, required=True, help="视频编号，单个数字或者以逗号、横杠分隔")
    parser.add_argument('--input_dir', type=str, default=os.path.join(CONFIG.PATH.DATASETS_PATH, 'needle/images'),
                        help="输入目录")
    parser.add_argument('--output_dir', type=str, default="segment_anns", help="输出目录")
    parser.add_argument('--num_videos_per_gpu', type=int, default=4, help="每个 GPU 处理的视频数量")
    parser.add_argument('--topn', type=int, default=5, help="筛选的掩码数量")
    args = parser.parse_args()
    run(parse_numbers(args.video_num), args.input_dir, args.output_dir, args.num_videos_per_gpu, args.topn)
