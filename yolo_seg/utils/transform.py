"""各种图片转换功能函数"""
import numpy as np

__all__ = [
    'crop_frame',
]


def crop_frame(frame, xyxy, crop_size=380):
    """
      功能：帧区域裁剪
    """
    height, width, _ = frame.shape
    x1, y1, x2, y2 = xyxy
    x_center, y_center = int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    # 扩展到crop_sizexcrop_size
    half_size = crop_size // 2
    x1 = x_center - half_size
    y1 = y_center - half_size
    x2 = x_center + half_size
    y2 = y_center + half_size
    
    # 边界检查
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    cropped_image = frame[y1:y2, x1:x2]
    
    # 如果切图大小不足crop_sizexcrop_size，进行补边
    if cropped_image.shape[0] < crop_size or cropped_image.shape[1] < crop_size:
        padded_image = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        padded_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
        cropped_image = padded_image
    
    # for debug: 观察裁剪的图片
    # fpath =os.path.join('./crop_images_for_classify/', f'{cnt}.jpg')
    # print(fpath)
    # cv2.imwrite(fpath, cropped_image)
    # cnt+=1
    return cropped_image
