import os
import json
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
import cv2

from needle_seg.utils import polygon_encoding


def get_bbox_from_mask(mask):
    # 找到轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取外接矩形框 (bounding box)
    if len(contours) == 0:
        return None
    bbox = cv2.boundingRect(contours[0])
    
    # bbox 是一个 (x, y, w, h) 元组，表示左上角坐标 (x, y) 和宽度w、高度h
    return bbox


# -------------------------------------
# 路径和文件
# -------------------------------------
DATASET_ROOT = "resources/datasets/needle-seg-full"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
PANOMASKS_DIR = os.path.join(DATASET_ROOT, "panomasks")
PANOPTIC_JSON = os.path.join(DATASET_ROOT, "panoptic_gt_VIPSeg.json")
CATEGORIES_JSON = os.path.join(DATASET_ROOT, "panoVIPSeg_categories.json")

# 加载类别定义
with open(CATEGORIES_JSON, 'r') as f:
    categories = json.load(f)

# 只保留thing类类别
categories_thing = [c for c in categories if c.get('isthing', 1) == 1]

# 建立categories的映射: {cat_id: cat_info}
cat_id_map = {c['id']: c for c in categories_thing}

# 加载panoptic标注
with open(PANOPTIC_JSON, 'r') as f:
    panoptic_data = json.load(f)

# panoptic_data 有结构:
# {
#   "videos": [...],
#   "annotations": [...],
#   "categories": [...]
# }

# 我们需要将其转换为COCO实例格式:
# COCO实例格式需要:
# "images": [{"id": int, "file_name": str, "height": int, "width": int}, ...]
# "annotations": [{"id": int, "image_id": int, "category_id": int, "segmentation": RLE, "bbox": [...], "area": int, "iscrowd": int}, ...]
# "categories": 和原来类似，只包含thing类

images_coco = []
annotations_coco = []
image_id_map = {}
ann_id = 1
img_id = 1


video_image_info = {}
for v in panoptic_data['videos']:
    video_id = v['video_id']
    for img_info in v['images']:
        # 以 (video_id, file_name) 作为key
        key = (video_id, img_info['file_name'])
        video_image_info[key] = img_info

# 遍历annotations中的video级数据
for v_anno in panoptic_data['annotations']:
    video_id = v_anno['video_id']
    for frame_anno in v_anno['annotations']:
        file_name = frame_anno['file_name']
        key = (video_id, file_name)
        if key not in video_image_info:
            print(f"Warning: {key} not in video_image_info")
            continue
        img_info = video_image_info[key]
        
        width = img_info['width']
        height = img_info['height']
        
        # 为images_coco添加条目
        images_coco.append({
            "id": img_id,
            "file_name": os.path.join(video_id, file_name.replace('png', 'jpg')),
            "width": width,
            "height": height
        })
        current_image_id = img_id
        img_id += 1
        
        # 加载对应的panomask图像
        panomask_path = os.path.join(PANOMASKS_DIR, video_id, file_name)
        if not os.path.exists(panomask_path):
            print(f"Panomask not found: {panomask_path}")
            continue
        
        panomask = np.array(Image.open(panomask_path), dtype=np.uint16)
        
        panomask[panomask != 0] = 1
        polygons = polygon_encoding(panomask, False)
        bbox = get_bbox_from_mask(panomask)
        # segments_info中每个实例都有 id(==mask值), category_id, iscrowd, area
        for seg in frame_anno['segments_info']:
            seg_id = seg["id"]
            cat_id = seg["category_id"]  # 注意该category_id要与categories.json对齐
            iscrowd = seg["iscrowd"]
            area = seg["area"]
            

            # 计算bbox
            # ys, xs = np.where(instance_mask)
            # x_min, x_max = xs.min(), xs.max()
            # y_min, y_max = ys.min(), ys.max()
            # bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
            #
            # # 编码为RLE
            # rle = mask_utils.encode(np.asarray(instance_mask, order='F'))
            # rle["counts"] = rle["counts"].decode('ascii')
            
            coco_cat_id = cat_id
            annotations_coco.append({
                "id": ann_id,
                "image_id": current_image_id,
                "category_id": coco_cat_id,
                "segmentation": [polygons],
                "area": int(area),
                "bbox": bbox,
                "iscrowd": iscrowd
            })
            ann_id += 1

# 最终COCO实例格式
instance_json = os.path.join(DATASET_ROOT, "vipseg_instance.json")
with open(instance_json, 'w') as f:
    json.dump({
        "images": images_coco,
        "annotations": annotations_coco,
        "categories": categories_thing
    }, f)

print(f"COCO instance format saved at {instance_json}")
