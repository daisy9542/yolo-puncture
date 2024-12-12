import os
import torch
import glob
import cv2
from torch.utils.data import Dataset


class NeedleSegVideoReader(Dataset):
    """
    每个 NeedleSegVideoReader 对象代表一个视频序列, 包含多帧图像。
    """
    
    def __init__(self, vid_name, frames, labels_dir):
        # vid_name: 视频编号（字符串）
        # frames: 当前视频的所有帧文件路径列表
        # labels_dir: 对应labels的目录路径
        # size: (H, W) 如需要resize，可在这里处理
        self.vid_name = vid_name
        self.frames = frames
        self.labels_dir = labels_dir
        # 可以根据需要定义调色板，这里用None
        self.palette = None
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        img_path = self.frames[idx]
        filename = os.path.basename(img_path)
        # 对应的txt标注
        lbl_path = os.path.join(self.labels_dir, filename.replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        
        need_resize = False
        
        # 解析txt标注
        segments_info = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.read().strip().split('\n')
                for obj_id, line in enumerate(lines):
                    parts = line.split()
                    class_index = int(parts[0])
                    coords = parts[1:]
                    polygon = []
                    for i in range(0, len(coords), 2):
                        x_norm = float(coords[i])
                        y_norm = float(coords[i + 1])
                        x_abs = x_norm * w
                        y_abs = y_norm * h
                        polygon.append((x_abs, y_abs))
                    
                    segments_info.append({
                        "id": obj_id + 1,
                        "category_id": class_index,
                        "isthing": 1,
                        "polygon": polygon
                    }
                    )
        
        # 构造info字典
        info = {
            'frame': [filename],
            'shape': (h, w),
            'need_resize': [need_resize],
            'is_rgb': [False],  # 如果需要根据图像特点调整
            'path_to_image': [img_path],
            'save': [True],  # 根据需要决定是否保存所有帧结果
            'json': None  # 我们没有json, 可设为None
        }
        
        # 将图像转换为tensor (C,H,W) 格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
        
        return {
            'rgb': image_tensor.unsqueeze(0),  # batch维度
            'mask': None,  # 没有现成的mask，这里设为None
            'info': info,
            'segments_info': segments_info
        }


class NeedleSegDetectionTestDataset:
    """
    将指定目录下的images和labels组织成与VIPSeg类似的视频数据集接口。
    """
    
    def __init__(self, root_dir, split='val'):
        # root_dir = /home/puncture/datasets/needle-seg-full
        self.root_dir = root_dir
        self.split = split
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.lbl_dir = os.path.join(root_dir, 'labels', split)
        
        # 找出所有图片
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
        # 根据文件名解析出 video_num, frame_num
        # 文件格式: [video_num]frame_[frame_num].jpg
        # 假设video_num和frame_num都是整数
        from collections import defaultdict
        vids_dict = defaultdict(list)
        
        for img_path in self.img_files:
            fname = os.path.basename(img_path)
            # 简单解析，如"44frame_10.jpg"
            # 可以用split或正则
            # 假设格式严格: 前面是video_num+字符串"frame_",然后是frame_num
            # e.g. "44frame_10.jpg" -> video_num=44, frame_num=10
            base = fname.replace('.jpg', '')
            # 以'frame_'拆分
            # base = "44frame_10"
            parts = base.split('frame_')
            video_num = parts[0]  # "44"
            # frame_num = parts[1] # "10" 不一定需要frame_num排序
            
            vids_dict[video_num].append(img_path)
        
        # 按视频排序
        self.videos = []
        for vnum, frames in vids_dict.items():
            frames = sorted(frames, key=lambda x: int(os.path.basename(x).split('frame_')[1].replace('.jpg', '')))
            self.videos.append((vnum, frames))
    
    def __len__(self):
        return len(self.videos)
    
    def get_datasets(self):
        # 原VIPSegDetectionTestDataset返回的是一个视频级dataset列表
        vid_datasets = []
        for vid_name, frames in self.videos:
            vid_ds = NeedleSegVideoReader(vid_name, frames, self.lbl_dir)
            vid_datasets.append(vid_ds)
        return vid_datasets
