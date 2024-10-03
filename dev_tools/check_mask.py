"""
检查筛选过后的 mask 是否仍然符合要求，不符合则删除该图片及对应的标签
按 `d` 删除该张不符合要求的图片，按 `enter` 继续下一张
"""
import os
import cv2
import numpy as np

from yolo_seg.utils.config import get_config

CONFIG = get_config()
DATASETS_DIR = CONFIG.PATH.DATASETS_PATH

def draw_polygon(image, points):
    height, width = image.shape[:2]
    points = (points * [width, height]).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

for path in ["train", "val"]:
    images_dir = f"{DATASETS_DIR}/needle-seg/images/{path}"
    labels_dir = f"{DATASETS_DIR}/needle-seg/labels/{path}"

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))

        image = cv2.imread(image_path)

        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                class_id = int(values[0])
                polygon_points = np.array([float(v) for v in values[1:]]).reshape(-1, 2)
                draw_polygon(image, polygon_points)

        cv2.imshow(f"Image: {image_file}", image)
        key = cv2.waitKey(0)

        if key == ord('d'):
            os.remove(image_path)
            os.remove(label_path)
            print(f"Deleted {image_file} and its label.")
        cv2.destroyAllWindows()