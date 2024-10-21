import os
from os import path
import re
import cv2
from PIL import Image
import numpy as np
import tempfile

from torch.utils.data.dataset import Dataset

__all__ = [
    'VideoReader',
    'sort_key',
]


class VideoReader(Dataset):
    """
    读取一个目录下的所有图片或读取某个视频文件，将视频文件拆分为图片帧，
    将视频文件拆分为多帧图片，命名格式为 {video_number}frame_{frame_number}.jpg。
    """
    
    # 支持的文件扩展名
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    VIDEO_EXTENSIONS = ['.mp4', '.avi']
    
    def __init__(self, images_path):
        self.images_path = images_path
        self.frames = []
        self.output_dir = None
        
        if path.isdir(self.images_path):
            self._load_image_directory(self.images_path)
        elif path.isfile(self.images_path) and self._is_video_file(self.images_path):
            self._process_video_file(self.images_path)
        else:
            raise ValueError("输入路径必须是一个图片目录或一个支持的视频文件。")
        
        # 对所有帧文件进行排序
        self.frames = sorted(self.frames, key=sort_key)
    
    def _is_video_file(self, filepath):
        """判断文件是否为支持的视频文件。"""
        _, ext = path.splitext(filepath)
        return ext.lower() in self.VIDEO_EXTENSIONS
    
    def _load_image_directory(self, directory):
        """加载目录中的所有图片文件。"""
        for filename in os.listdir(directory):
            file_path = path.join(directory, filename)
            if path.isfile(file_path):
                _, ext = path.splitext(filename)
                if ext.lower() in self.IMAGE_EXTENSIONS:
                    self.frames.append(filename)
        print(f"Loaded {len(self.frames)} images from '{directory}'.")
    
    def _process_video_file(self, video_path):
        """处理视频文件，将其拆分为多帧图片并保存。"""
        video_filename = path.basename(video_path)
        match = re.search(r'video(\d+)', video_filename, re.IGNORECASE)
        if not match:
            video_number = ""
        else:
            video_number = match.group(1)
        
        self.output_dir = tempfile.mkdtemp()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to read file '{video_filename}'。")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"{video_number}frame_{frame_count}.jpg"
            frame_path = path.join(self.output_dir, frame_filename)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.save(frame_path)
            frame_count += 1
        cap.release()
        
        for file in os.listdir(self.output_dir):
            if re.match(rf'^{re.escape(video_number)}frame_\d+\.jpg$', file, re.IGNORECASE):
                self.frames.append(file)
        
        print(f"Loaded {len(self.frames)} frames from '{video_path}'.")
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.output_dir:
            im_path = path.join(self.output_dir, frame)
        else:
            im_path = path.join(self.images_path, frame)
        img = Image.open(im_path).convert('RGB')
        img = np.array(img)
        return img, im_path
    
    def __len__(self):
        return len(self.frames)


def sort_key(filename):
    """
    定义用于排序的键函数。根据文件名中的数字部分进行排序。
    例如，'12frame_1.jpg' 会被排序在 '12frame_2.jpg' 前面。
    """
    match = re.findall(r'\d+', filename)
    return list(map(int, match)) if match else [0]
