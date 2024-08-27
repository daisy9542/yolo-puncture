import numpy as np
import cv2

KEY_FRAME = {
    # video_num: [start_frame, end_frame]
    1: [122, 165],
    2: [44, 94],
    3: [0, 23],
    4: [73, 113],
    5: [27, 49],
    6: [25, 64],
    7: [14, 61],
    8: [141, 180],
    9: [12, 27],
    10: [7, 23],
    11: [3, 19],
    12: [9, 34],
    13: [9, 25],
    14: [29, 57],
    15: [30, 56],
    16: [62, 81],
    17: [151, 165],
    18: [111, 128],
    19: [60, 92],
}

FRAME_OFFSET = 20


def id_assign(video_num, frame_num):
    return int(video_num * 1e6 + frame_num)


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
