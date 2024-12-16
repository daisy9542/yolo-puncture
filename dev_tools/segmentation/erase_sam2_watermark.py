"""
去除 SAM2 Demo 的水印
"""
import cv2
from tqdm import tqdm
import os
import argparse

video_dir = "resources/datasets/sam2-seg_副本/masks"

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Erase watermark from SAM2 Demo videos')
    args.add_argument('source', type=str, help='Path to the source video directory')
    args.add_argument('target', type=str, help='Path to the target video directory')
    args = args.parse_args()
    video_files = [video_file for video_file in os.listdir(video_dir) if video_file.endswith(".mp4")]
    
    for video_file in tqdm(video_files):
        if not video_file.endswith(".mp4"):
            continue
        # 视频路径
        input_video_path = os.path.join(args.source, video_file)
        output_video_path = os.path.join(args.target, video_file)
        
        # 打开视频文件
        print(f"Processing {input_video_path}")
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("无法打开视频文件")
            exit()
        
        # 获取视频属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 原视频的帧率
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 原视频总帧数
        
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*"h264")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        # 设置黑色覆盖区域的尺寸
        black_region_height = 100  # 距离底部100像素
        black_region_width = 400  # 距离右侧400像素
        
        frame_processed = 0
        
        # 处理每一帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 在右下角覆盖一个黑色矩形区域
            frame[frame_height - black_region_height:frame_height, frame_width - black_region_width:frame_width] = 0
            
            # 写入处理后的视频帧
            out.write(frame)
            frame_processed += 1
        
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
