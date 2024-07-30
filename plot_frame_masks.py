import cv2
import numpy as np
import json
import os
from dev_tools.small_tools import KEY_FRAME


def rle_encoding(binary_segment):
    """将二值化的掩码数组转换为 RLE 编码。"""
    pixels = binary_segment.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


def polygon_encoding(binary_segment, normalize=True):
    """将二值掩码转换为多边形格式，可选择归一化坐标。"""
    binary_segment = binary_segment.astype(np.uint8)
    # 查找轮廓
    contours, _ = cv2.findContours(binary_segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        contour = contour.reshape(-1, 2).tolist()
        
        # 如果需要归一化
        if normalize:
            # 获取图像的宽度和高度
            height, width = binary_segment.shape[:2]
            # 归一化坐标
            contour = [(round(min(1.0, max(0.0, x / width)), 6),
                        round(min(1.0, max(0.0, y / height)), 6))
                       for x, y in contour]
        contour = [coord for point in contour for coord in point]
        polygons.extend(contour)
    
    return polygons


def show_image_with_masks(image, anns, text):
    img = image.copy()
    for index, ann in enumerate(anns):
        m = ann['segmentation']
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)  # 使用绿色绘制轮廓
        
        # 获取遮罩的中心位置
        y_coords, x_coords = np.where(m)
        y_center = int(np.mean(y_coords))
        x_center = int(np.mean(x_coords))
        
        # 在遮罩中心位置附近添加序号
        cv2.putText(img, str(index + 1), (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 在图片左上角显示当前索引
    cv2.putText(img, f"Index: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Press 's' to skip, 'q' to quit.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return img


def mouse_callback(event, x, y, flags, param):
    global selected_index, masks_sorted, selected_mask, move_to_next
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float('inf')
        for idx, mask in enumerate(masks_sorted):
            m = mask['segmentation']
            y_coords, x_coords = np.where(m)
            y_center = float(np.mean(y_coords))
            x_center = float(np.mean(x_coords))
            dist = np.sqrt((x - x_center)**2 + (y - y_center)**2)
            if dist < min_dist:
                min_dist = dist
                selected_index = idx
        
        selected_mask = masks_sorted[selected_index]
        polygon = polygon_encoding(selected_mask['segmentation'])
        polygon.insert(0, 0)  # 标签中第一个数字表示类别，其他数字表示多边形坐标
        print(f"Selected mask index: {selected_index + 1} with area {selected_mask['area']}")
        print(f"Polygon coordinates: {polygon}")
        
        move_to_next = True


def process_and_save_masks(video_num, start_frame=0, end_frame=-1):
    global masks_sorted, selected_index, selected_mask, move_to_next
    all_masks = np.load(f"segment_anns/video{video_num}.npy", allow_pickle=True)
    end_frame = len(all_masks) - 1 if end_frame == -1 else end_frame
    start_frame = max(0, start_frame)
    end_frame = min(len(all_masks), end_frame)
    exception_files = []
    
    prev_area = None
    os.makedirs(f"annotations/images", exist_ok=True)
    os.makedirs(f"annotations/labels", exist_ok=True)
    for idx, masks in enumerate(all_masks):
        if idx < start_frame or idx > end_frame:
            continue
        image_path = f"images/train/{video_num}frame_{idx}.jpg"
        image = cv2.imread(image_path)
        
        if len(masks) == 0:
            exception_files.append(image_path)
            continue
        
        masks_sorted = sorted(masks, key=lambda x: abs(x['area'] - (prev_area if prev_area is not None else 0)))
        image_with_masks = show_image_with_masks(image, masks_sorted, f"Frame: {idx} ({start_frame}-{end_frame})")
        
        cv2.imshow('Image with Masks', image_with_masks)
        cv2.setMouseCallback('Image with Masks', mouse_callback)
        
        move_to_next = False
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                exception_files.append(image_path)
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif move_to_next:
                if selected_mask:
                    output_image_path = f"annotations/images/{video_num}frame_{idx}.jpg"
                    cv2.imwrite(output_image_path, image)
                    
                    label_path = f"annotations/labels/{video_num}frame_{idx}.txt"
                    with open(label_path, "w") as f:
                        polygon = polygon_encoding(selected_mask['segmentation'])
                        polygon.insert(0, 0)
                        f.write(" ".join(map(str, polygon)))
                    
                    prev_area = selected_mask['area']
                break
        
        cv2.destroyAllWindows()
    
    os.makedirs("annotations/exceptions", exist_ok=True)
    with open(f"annotations/exceptions/video{video_num}_exceptions.json", 'w') as f:
        print(f"Save to annotations/exceptions/video{video_num}_exceptions.json ...")
        json.dump(exception_files, f)


if __name__ == '__main__':
    video_num = 7
    start_frame, end_frame = KEY_FRAME[video_num]
    process_and_save_masks(video_num, start_frame, end_frame)
