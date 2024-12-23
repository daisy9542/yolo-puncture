"""
将 Mask2Former 的推理结果 json 文件转换为 掩码图片。
"""
import os
import json
import cv2
import numpy as np
from pycocotools import mask as coco_mask
from collections import defaultdict
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask2Former 掩码转换')
    parser.add_argument('--train-json', type=str,
                        default='resources/datasets/needle-seg-full/vipseg_instance_test.json',
                        help='训练数据 JSON 文件')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    args = parser.parse_args()
    inference_json = os.path.join(args.output_dir, 'coco_instances_results.json')
    
    # 读取训练文件
    with open(args.train_json, 'r') as f:
        train_data = json.load(f)
    
    # 读取推理结果文件
    with open(inference_json, 'r') as f:
        inference_data = json.load(f)
    
    # 将训练数据的图片信息（image_id -> file_name）存储在字典中
    image_info = {image['id']: image['file_name'] for image in train_data['images']}
    
    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建用于存储每个视频的掩码图像
    video_masks = defaultdict(list)
    
    # 用于存储每个 image_id 对应的推理结果（按照 image_id 分组）
    image_results = defaultdict(list)
    
    # 处理每个推理结果
    for result in inference_data:
        image_id = result['image_id']
        # 获取对应的文件名
        file_name = image_info.get(image_id, None)
        if file_name is None:
            print(f"Image ID {image_id} not found in training data.")
            continue  # 如果没有找到对应的图片ID，则跳过
        
        # 从推理结果获取分割掩码（segmentation）和分数（score）
        segmentation = result['segmentation']
        score = result['score']
        
        # 将每个推理结果按 image_id 分组
        image_results[image_id].append({
            'score': score,
            'segmentation': segmentation
        })
    
    # 处理每个 image_id 的结果，选择分数最高的掩码
    for image_id, results in image_results.items():
        # 按照分数排序，选择分数最高的
        best_result = max(results, key=lambda x: x['score'])
        segmentation = best_result['segmentation']
        
        # 获取文件名
        file_name = image_info[image_id]
        video_id = file_name.split('/')[0]
        image_name = file_name.split("/")[-1].replace(".jpg", ".png")
        
        # 获取掩码的尺寸（size）
        mask_size = segmentation['size']
        
        # 将分割结果解码成掩码
        mask = coco_mask.decode(segmentation)
        
        # 创建视频文件夹路径
        video_output_dir = os.path.join(args.output_dir, 'pan_pred', video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # 在全黑的图像上绘制掩码
        black_mask_image = np.zeros((mask_size[0], mask_size[1], 3), dtype=np.uint8)  # 创建全黑图像 (RGB)
        black_mask_image[mask == 1] = [0, 0, 255]  # 掩码区域用红色表示
        
        # 保存掩码图像
        mask_output_path = os.path.join(video_output_dir, image_name)
        cv2.imwrite(mask_output_path, black_mask_image)
        
        print(f"Saved mask image at {mask_output_path}")
    
    print("Processing complete.")
