"""
使用 YOLO 模型对指定的多个视频目录进行图像分割
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import argparse
import json

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

obj_id = 10000


def process_video(video_dir, model, output_dir, conf_thresh=0.25, batch_size=1):
    """
    对单个视频目录下的所有图片进行分割，并保存结果
    """
    global obj_id
    video_name = os.path.basename(video_dir.rstrip("/"))
    video_output_dir = os.path.join(output_dir, "pan_pred", video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    video_annotations = {"video_id": video_name, "annotations": []}
    
    # 使用 YOLO 直接预测目录
    results = model.predict(source=video_dir, conf=conf_thresh, retina_masks=True, batch=batch_size, device=device)
    torch.cuda.empty_cache()
    
    # 将结果按文件名排序
    results = sorted(results, key=lambda x: os.path.basename(x.path))
    
    for result in results:
        image_file = os.path.basename(result.path)  # 从结果中获取文件名
        h, w = result.orig_shape
        pred_boxes = result.boxes.cpu().numpy()
        
        # 初始化掩码图像为全零
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        segments_info = []
        
        if len(pred_boxes.cls) > 0:
            # 若检测到多个物体，取置信度最大的
            best_conf_idx = np.argmax(pred_boxes.conf)
            mask = result.masks.data[best_conf_idx].cpu().numpy()  # 获取对应的掩码
            
            mask_rgb[mask == 1] = (0, 0, 255)
            
            confidence = pred_boxes.conf[best_conf_idx]
            category_id = int(pred_boxes.cls[best_conf_idx])
            bbox = pred_boxes.xywh[best_conf_idx]
            segments_info.append({
                "id": obj_id,
                "category_id": category_id + 1
            })
            obj_id += 1
        
        # 保存分割掩码图像
        output_img_path = os.path.join(video_output_dir, os.path.splitext(image_file)[0] + '.png')
        cv2.imwrite(output_img_path, mask_rgb)  # 保存掩码图像
        
        # 保存 JSON annotation
        video_annotations["annotations"].append({
            "file_name": image_file,
            "segments_info": segments_info
        })
    
    return video_annotations


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='YOLO segmentation')
    args.add_argument('--weight', type=str, help='weight path')
    args.add_argument('--source', type=str, help='txt file containing video directories')
    args.add_argument('--output', type=str, help='output path')
    args.add_argument('--batch', type=int, default=1, help='batch size')
    args.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    
    args = args.parse_args()
    model = YOLO(args.weight, task='segmentation')
    
    # 读取 source txt 文件
    with open(args.source, 'r') as f:
        video_dirs = [line.strip() for line in f.readlines()]
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化全局 JSON 输出
    global_annotations = {"annotations": []}
    
    # 遍历 txt 中指定的每个视频目录
    for video_dir in video_dirs:
        video_path = os.path.join(os.path.dirname(args.source), video_dir)
        if os.path.isdir(video_path):
            print(f"Processing video: {video_dir}")
            video_annotations = process_video(video_path, model, output_dir, conf_thresh=args.conf,
                                              batch_size=args.batch
                                              )
            global_annotations["annotations"].append(video_annotations)
        else:
            print(f"Warning: {video_dir} is not a valid directory. Skipping.")
    
    # 保存全局 pred.json 文件
    json_output_path = os.path.join(output_dir, "pred.json")
    with open(json_output_path, "w") as json_file:
        json.dump(global_annotations, json_file, indent=4)
    
    print(f"Segmentation masks and JSON results saved to: {output_dir}")
