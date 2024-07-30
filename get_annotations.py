import os
import re
import cv2
import numpy as np
import argparse
from functools import cmp_to_key
from tqdm import tqdm
from utils.config import get_config
from utils.segment_everything import segment
from utils.mask_tools import filter_masks

CONFIG = get_config()


def compare_filenames(f1, f2):
    # 分割文件名，获取视频编号和帧编号
    pattern = re.compile(r'.*/(\d+)frame_(\d+)\.jpg')
    match1 = pattern.match(f1)
    match2 = pattern.match(f2)
    video1, frame1 = map(int, match1.groups())
    video2, frame2 = map(int, match2.groups())
    
    # 首先按视频编号排序，如果视频编号相同，则按帧编号排序
    if video1 < video2:
        return -1
    elif video1 > video2:
        return 1
    else:
        if frame1 < frame2:
            return -1
        elif frame1 > frame2:
            return 1
    return 0


def process_and_save_images(input_dir, output_dir, video_nums):
    # 获取所有图片文件
    files_to_process = []
    for video_num in video_nums:
        print(f"Processing video {video_num} ...")
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.startswith(f"{video_num}frame") and file.endswith(('.jpg', '.jpeg', '.png')):
                    files_to_process.append(os.path.join(root, file))
        
        output_file_path = os.path.join(output_dir, f"video{video_num}.npy")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        video_masks = []
        sorted_files = sorted(files_to_process, key=cmp_to_key(compare_filenames))
        for input_file_path in tqdm(sorted_files, desc="Processing images"):
            # 读取图像
            image = cv2.imread(input_file_path)
            
            # 处理图像，获得 masks
            masks = segment(image)
            masks = filter_masks(masks, 5)
            
            video_masks.append(masks)
        # 保存masks到输出目录
        np.save(output_file_path, video_masks)


def parse_numbers(s):
    if s.isdigit():
        return [s]
    else:
        return [item for item in s.split(',')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="segment annotations")
    parser.add_argument('--video_num', type=str, required=True, help="视频编号，单个数字或者以逗号分隔")
    input_dir = os.path.join(CONFIG.PATH.DATASETS_PATH, 'needle/images')
    output_dir = "segment_anns"
    args = parser.parse_args()
    process_and_save_images(input_dir, output_dir, parse_numbers(args.video_num))
