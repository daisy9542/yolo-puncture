import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(frame_shape, anns, x_offset=0, y_offset=0):
    """
    生成一个mask数组，用于cv2.addWeighted函数。

    Args:
    - image_shape: 原始图像的形状 (height, width, channels)。
    - anns: 包含遮罩信息的列表。
    - x_offset, y_offset: 遮罩位置的偏移量，用于将局部遮罩放置在全图的正确位置。

    Returns:
    - mask: 生成的mask数组。
    """
    if len(anns) == 0:
        return np.zeros(frame_shape, dtype=np.uint8)
    
    height, width, _ = frame_shape
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for ann in anns:
        segmentation = ann['segmentation']
        color_mask = np.random.randint(0, 255, (3,), dtype=int)  # RGB color
        for i in range(segmentation.shape[0]):
            for j in range(segmentation.shape[1]):
                if segmentation[i, j]:
                    mask[y_offset + i, x_offset + j] = color_mask
        # 获取遮罩的中心位置
        y_coords, x_coords = np.where(segmentation)
        y_center = np.mean(y_coords) + y_offset
        x_center = np.mean(x_coords) + x_offset
        
        # 在遮罩中心位置附近添加面积标签
        area_text = f"{ann['area']:.1f}"
        cv2.putText(mask, area_text, (int(x_center), int(y_center)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return mask


def segment(image, model_type="vit_l", device="cuda"):
    """
    vit_h: ViT-H SAM model, sam_vit_h_4b8939.pth (huge)
    vit_l: ViT-L SAM model, sam_vit_l_0b3195.pth (large)
    vit_b: ViT-B SAM model, sam_vit_b_01ec64.pth (base)
    """
    sam_checkpoint = f"weights/sam/sam_{model_type}.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        points_per_batch=16,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )
    masks = mask_generator.generate(image)
    return masks
