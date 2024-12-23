"""
根据相应的 train.txt、val.txt 和 test.txt 文件来划分 json 文件。
如果有的话，也划分 COCO 格式标注文件。
"""
import os
import json

ROOT_DIR = 'resources/datasets/needle-seg-full'


def split_vipseg_json():
    # 读取原始的 JSON 文件
    input_file = os.path.join(ROOT_DIR, 'panoptic_gt_VIPSeg.json')
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 使用字典按 video_id 索引注释，避免重复遍历
    annotations_dict = {}
    for ann in data['annotations']:
        video_id = ann['video_id']
        if video_id not in annotations_dict:
            annotations_dict[video_id] = []
        annotations_dict[video_id].append(ann)
    
    # 按照 split 文件进行处理
    for split in ['train', 'val', 'test']:
        # 读取视频 ID 列表
        with open(os.path.join(ROOT_DIR, f'{split}.txt'), 'r') as f:
            split_videos = {line.strip().split('/')[-1] for line in f.readlines()}  # 使用 strip() 去除换行符
        
        split_data = {
            'videos': [],
            'annotations': [],
            'categories': data['categories']
        }
        
        # 遍历视频数据
        for video in data['videos']:
            if video['video_id'] in split_videos:
                split_data['videos'].append(video)
                # 将对应的注释添加到 `annotations` 列表
                split_data['annotations'].extend(annotations_dict.get(video['video_id'], []))
        
        # 保存拆分后的数据
        output_file = os.path.join(ROOT_DIR, f'panoptic_gt_VIPSeg_{split}.json')
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=4)
        
        print(f'Finished processing {split}.txt')


def split_coco_json():
    coco_json = os.path.join(ROOT_DIR, 'vipseg_instance.json')
    train_txt = os.path.join(ROOT_DIR, 'train.txt')
    val_txt = os.path.join(ROOT_DIR, 'val.txt')
    test_txt = os.path.join(ROOT_DIR, 'test.txt')
    if not os.path.exists(coco_json):
        return
    
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # 读取train.txt、val.txt、test.txt中的图像文件名
    def read_txt(file_path):
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
        return set(lines)
    
    train_videos = read_txt(train_txt)
    val_videos = read_txt(val_txt)
    test_videos = read_txt(test_txt)
    
    # 初始化拆分数据结构
    splits = {
        'train': {'images': [], 'annotations': []},
        'val': {'images': [], 'annotations': []},
        'test': {'images': [], 'annotations': []}
    }
    
    # 将图像分配到各个拆分
    for split_name, video_set in zip(['train', 'val', 'test'],
                                     [train_videos, val_videos, test_videos]):
        for video in video_set:
            # 获取当前视频对应的所有图像文件名
            for image in coco_data['images']:
                if video in image['file_name'].split("/"):
                    splits[split_name]['images'].append(image)
    
    # 创建图像ID到拆分的映射
    image_id_to_split = {}
    for split_name in splits:
        for img in splits[split_name]['images']:
            image_id_to_split[img['id']] = split_name
    
    # 将标注分配到各个拆分
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        split_name = image_id_to_split.get(image_id)
        if split_name:
            splits[split_name]['annotations'].append(ann)
        else:
            print(f"警告: 标注的 image_id {image_id} 没有对应的拆分。")
    
    # 准备并保存拆分后的JSON文件
    for split_name in splits:
        output_json_path = os.path.join(ROOT_DIR, f'vipseg_instance_{split_name}.json')
        
        split_json = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'images': splits[split_name]['images'],
            'annotations': splits[split_name]['annotations'],
            'categories': coco_data['categories']
        }
        # 写入拆分后的JSON文件
        with open(output_json_path, 'w') as f:
            json.dump(split_json, f)
        print(f"已保存 {split_name} 拆分到 {output_json_path}")


if __name__ == '__main__':
    split_vipseg_json()
    split_coco_json()
