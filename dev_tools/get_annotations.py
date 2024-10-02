import os
import cv2
import argparse
import pickle
from tqdm import tqdm
from yolo_seg.utils.config import get_config
from yolo_seg.utils.segment_anything import segment
from yolo_seg.utils.mask_tools import filter_masks
from dev_tools.toolbox import polygon_encoding

CONFIG = get_config()


def process_video(video_files, output_dir, topn):
    for video_file in tqdm(video_files, desc="Processing videos", unit="video"):
        video_capture = cv2.VideoCapture(video_file)
        video_num = os.path.basename(video_file).split('.')[0]  # 从文件名中获取视频编号
        
        output_file_path = os.path.join(output_dir, f"{video_num}.pkl")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        video_masks = {}
        
        frame_num = 0
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
        for _ in tqdm(range(total_frames), desc=f"Processing {video_num}", unit="frame", leave=False):
            ret, frame = video_capture.read()  # 逐帧读取视频
            if not ret:
                break  # 读取到视频末尾时退出
            
            masks = segment(frame)
            masks = filter_masks(masks, topn)
            
            polygon_masks = []
            for mask in masks:
                polygon_masks.append({
                    "segmentation": polygon_encoding(mask["segmentation"]),
                    "bbox": mask["bbox"]
                })
            
            video_num = video_num.replace("video", "")
            filename = f"{video_num}frame_{frame_num}.jpg"
            video_masks[filename] = polygon_masks
            frame_num += 1
        
        video_capture.release()  # 释放视频资源
        pickle.dump(video_masks, open(output_file_path, 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="segment annotations from video")
    parser.add_argument('--video_files', type=str, required=True, help="视频文件路径，多个文件以逗号分隔")
    parser.add_argument('--output_dir', type=str, default="segment_anns", help="输出目录")
    parser.add_argument('--topn', type=int, default=15, help="筛选的掩码数量")
    args = parser.parse_args()
    
    video_files = []
    if os.path.isdir(args.video_files):
        for root, dirs, files in os.walk(args.video_files):
            for file in files:
                if file.endswith(".mp4"):
                    video_files.append(os.path.join(root, file))
    else:
        video_files.append(args.video_files)
    process_video(video_files, args.output_dir, args.topn)
