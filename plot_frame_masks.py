"""
为视频中的帧图像提供交互式的掩码标注和保存功能，
通过点击图像来选择目标区域，并将该区域的掩码保存下来
"""
import cv2
import numpy as np
import pickle
import os
from dev_tools.toolbox import KEY_FRAME

CONF_DIS = 20  # 距离置信度阈值，当前 masks 中任意 mask 距离上一次选中的点均大于这个值则直接跳过


def normalize_to_pixel_coordinates(normalized_coords, img_width, img_height):
    """将归一化坐标转换为像素坐标。"""
    pixel_coords = []
    for i in range(0, len(normalized_coords), 2):
        x = int(float(normalized_coords[i]) * img_width)
        y = int(float(normalized_coords[i + 1]) * img_height)
        pixel_coords.append((x, y))
    return pixel_coords


def show_image_with_masks(image, anns, label):
    img = image.copy()
    height, width, _ = img.shape
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for idx, ann in enumerate(anns):
        coords = ann["segmentation"]
        x, y, w, h = ann["bbox"]
        pixel_coords = normalize_to_pixel_coordinates(coords, width, height)
        points = np.array(pixel_coords, np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], isClosed=True, color=colors[idx % len(colors)], thickness=1)
        
        x_center = int(x + w / 2)
        y_center = int(y + h / 2)
        
        # 在遮罩中心位置附近添加序号
        cv2.putText(img, str(idx + 1), (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colors[idx % len(colors)], 2)
    
    # 在图片左上角显示当前索引
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, "Press 's' to skip, 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return img


def mouse_callback(event, point_x, point_y, flags, param):
    global anns, selected_index, selected_mask, move_to_next, prev_click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float('inf')
        prev_click_point = (point_x, point_y)
        for idx, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            x_center = x + w / 2
            y_center = y + h / 2
            dist = np.sqrt((point_x - x_center)**2 + (point_y - y_center)**2)
            if dist < min_dist:
                min_dist = dist
                selected_index = idx
        
        selected_mask = anns[selected_index]["segmentation"]
        
        move_to_next = True


def process_and_save_masks(video_num):
    global anns, selected_index, selected_mask, move_to_next, prev_click_point
    with open(f"segment_anns/video{video_num}.pkl", "rb") as f:
        all_masks = pickle.load(f)
    
    prev_click_point = None
    os.makedirs(f"annotations/images", exist_ok=True)
    os.makedirs(f"annotations/labels", exist_ok=True)
    
    for img_name, anns in all_masks.items():
        frame_num = int(img_name.split('_')[-1].split('.')[0])
        if img_name.startswith("video"):
            img_name = img_name.replace("video", "").replace("_", "", 1)
        image_path = f"images/{img_name}"
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        skip_image = True
        if prev_click_point is None:
            skip_image = False
        else:
            for ann in anns:
                x, y, w, h = ann["bbox"]
                center_x, center_y = (x + w / 2, y + h / 2)
                if prev_click_point:
                    dist = np.sqrt((center_x - prev_click_point[0])**2 + (center_y - prev_click_point[1])**2)
                    if dist <= CONF_DIS:
                        skip_image = False
                        break
        if skip_image:
            continue
        
        start_frame, end_frame = KEY_FRAME[video_num]
        image_with_masks = show_image_with_masks(
            image, anns,
            f"Video {video_num} frame {frame_num} ({start_frame}-{end_frame})")
        
        cv2.imshow('Image with Masks', image_with_masks)
        cv2.setMouseCallback('Image with Masks', mouse_callback)
        
        move_to_next = False
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif move_to_next:
                if selected_mask:
                    output_image_path = f"annotations/images/{video_num}frame_{frame_num}.jpg"
                    cv2.imwrite(output_image_path, image)
                    
                    label_path = f"annotations/labels/{video_num}frame_{frame_num}.txt"
                    selected_mask.insert(0, 0)
                    with open(label_path, "w") as f:
                        f.write(" ".join(map(str, selected_mask)))
                break
        
        cv2.destroyAllWindows()


# 点击的时候尽量点中心点，也就是数字显示的位置，因为会根据点击的坐标来过滤不合适的图片的 masks
if __name__ == '__main__':
    video_num = 14
    # for video_num in range(1, 20):
    start_frame, end_frame = KEY_FRAME[video_num]
    process_and_save_masks(video_num)
