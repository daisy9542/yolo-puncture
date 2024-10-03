"""
检查筛选过后的 mask 是否仍然符合要求，不符合则删除该图片及对应的标签。
按 `d` 删除该张不符合要求的图片，按 `enter` 继续下一张。
"""
import os
import cv2
import numpy as np

from toolbox import sort_by_filename


def draw_polygon(image, points):
    height, width = image.shape[:2]
    points = (points * [width, height]).reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)


if __name__ == '__main__':
    images_dir = f"resources/annotations/images"
    labels_dir = f"resources/annotations/labels"
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')],
                         key=sort_by_filename)
    i = 0
    total = len(image_files)
    
    while i < total:
        print(i, image_files[i])
        image_file = image_files[i]
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))
        
        image = cv2.imread(image_path)
        
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                class_id = int(values[0])
                polygon_points = np.array([float(v) for v in values[1:]]).reshape(-1, 2)
                draw_polygon(image, polygon_points)
        
        cv2.putText(image, "Press 'enter': skip, 'd': delete image, 'p': previous image",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow(f"Image: {image_file}", image)
        key = cv2.waitKey(0)
        
        if key == ord('d'):
            os.remove(image_path)
            os.remove(label_path)
        elif key == ord('p'):
            if i > 0:
                i -= 1
            continue
        i += 1
        cv2.destroyAllWindows()
