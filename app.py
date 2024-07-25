import gradio as gr
import cv2
import tempfile

from PIL import Image

from ultralytics import YOLOv10
import numpy as np
import torch
import io
import matplotlib.pyplot as plt
from utils.segment_everything import show_anns, segment

INIT_SHAFT_LEN = 20  # 针梗的实际长度，单位为毫米
MOVE_THRESHOLD = 2  # 针梗移动的阈值，单位为毫米
CONFIRMATION_FRAMES = 5  # 连续几帧确认像素比例和插入状态
OUT_EXPAND = 50  # 输出图像感兴趣区域的扩展像素数


def yolov10_inference(image, video,
                      yolo_model_id,
                      segment_model_id,
                      classify_model_id,
                      image_size,
                      yolo_conf_threshold,
                      classify_conf_threshold):
    if yolo_model_id.endswith("pt"):
        model = YOLOv10(f'{yolo_model_id}')
    else:
        model = YOLOv10.from_pretrained(f'{yolo_model_id}/', local_files_only=True)
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=yolo_conf_threshold)
        annotated_image = results[0].plot()
        
        image_darray = np.array(image)
        masks = segment(image_darray, segment_model_id)
        masks = filter_masks(masks, (0, 0, image_darray.shape[1], image_darray.shape[0]))
        print(masks)
        if len(masks) > 0:
            img_with_masks = draw_masks_on_image(image, masks)
        else:
            img_with_masks = image
        return img_with_masks, None
    else:
        video_path = tempfile.mktemp(suffix=".mp4")
        
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_video_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        pixel_len_arr = []  # 视频中针梗的长度，以像素为单位
        insert_counter = 0  # 连续插入计数器
        insert_spec_counter = 0  # 判断插入皮肤超出长度计数器
        inserted = False  # 是否插入皮肤（只判断初始固定距离）
        insert_start, insert_spec_end = None, None  # 记录插入皮肤的开始和指定结束时间
        insert_start_frame, insert_spec_end_frame = None, None  # 记录插入皮肤的开始和指定结束所在帧
        spec_insert_speed = None  # 插入皮肤指定长度的速度
        speed_clac_compute = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = np.zeros_like(frame)
            roi_mask = np.zeros_like(frame)
            
            results = model.predict(source=frame, imgsz=image_size, conf=yolo_conf_threshold)
            pred_boxes = results[0].boxes
            
            if len(pred_boxes.cls) > 0:
                # 若检测到多个物体，取置信度最大的
                best_conf_idx = torch.argmax(pred_boxes.conf)
                cls, conf = int(pred_boxes.cls[best_conf_idx]), float(pred_boxes.conf[best_conf_idx])
                name = results[0].names[cls]
                xyxy = pred_boxes.xyxy[best_conf_idx].squeeze()
                x1, y1, x2, y2 = map(int, xyxy.cpu().numpy())
                shaft_pixel_len = (x2 - x1) * (y2 - y1)  # 针梗的像素长度
                
                # 针梗分割
                if not speed_clac_compute:
                    height, width, _ = frame.shape
                    x1 = max(0, x1 - OUT_EXPAND)
                    y1 = max(0, y1 - OUT_EXPAND)
                    x2 = min(width, x2 + OUT_EXPAND)
                    y2 = min(height, y2 + OUT_EXPAND)
                    roi = rgb_frame[y1:y2, x1:x2]
                    masks = segment(roi, segment_model_id)
                    masks = filter_masks(masks, (x1, y1, x2, y2))
                    # TODO: masks 长度可能为 0
                    if len(masks) == 0:
                        continue
                    mask = show_anns(frame.shape, masks, x1, y1)
                    _, _, w, h = masks[0]['bbox']
                    shaft_pixel_len = np.sqrt(w**2 + h**2)
                
                if cls == 0 and not inserted and shaft_pixel_len is not None:
                    pixel_len_arr.append(shaft_pixel_len)
                    if len(pixel_len_arr) > CONFIRMATION_FRAMES:
                        pixel_len_arr.pop(0)
                if cls == 1 and len(pixel_len_arr) == 0:
                    # 第一帧就检测到插入皮肤的情况
                    pixel_len_arr.append(shaft_pixel_len)
                actual_len = INIT_SHAFT_LEN if cls == 0 else (
                        INIT_SHAFT_LEN * shaft_pixel_len / (sum(pixel_len_arr) / len(pixel_len_arr)))
                
                # 判断是否开始插入皮肤
                if cls == 1 and not inserted and not speed_clac_compute:
                    insert_counter += 1
                    if insert_counter >= CONFIRMATION_FRAMES:
                        inserted = True
                        insert_start = cap.get(cv2.CAP_PROP_POS_MSEC)
                        insert_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                # 判断是否插入皮肤达到指定长度
                if cls == 1 and inserted and actual_len <= INIT_SHAFT_LEN - MOVE_THRESHOLD:
                    insert_spec_counter += 1
                    if insert_spec_counter >= CONFIRMATION_FRAMES:
                        insert_spec_counter = 0
                        insert_counter = 0
                        inserted = False
                        speed_clac_compute = True
                        insert_spec_end = cap.get(cv2.CAP_PROP_POS_MSEC)
                        spec_insert_speed = 1000 * MOVE_THRESHOLD / (insert_spec_end - insert_start)
                        insert_spec_end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                
                if speed_clac_compute:
                    label = f"{name} {conf:.2f} {spec_insert_speed:.2f}mm/s"
                else:
                    label = f"{name} {conf:.2f} {actual_len:.2f} {shaft_pixel_len:.2f}"
                
                roi_mask = create_roi_mask(frame.shape, x1, y1, x2, y2, label)
            # annotated_frame = results[0].plot(label_content=label, best_conf_idx=best_conf_idx)
            combined_frame = cv2.addWeighted(frame, 1, mask, 1, 0)
            combined_frame = cv2.addWeighted(combined_frame, 1, roi_mask, 1, 0)
            out.write(combined_frame)
        
        cap.release()
        out.release()
        print("Start: ", insert_start_frame, " End: ", insert_spec_end_frame)
        
        return None, output_video_path


def draw_masks_on_image(image, masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    # show_anns(np.array(image).shape,masks)
    if len(masks) == 0:
        return
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        # 获取遮罩的中心位置
        y_coords, x_coords = np.where(m)
        y_center = np.mean(y_coords)
        x_center = np.mean(x_coords)
        
        # 在遮罩中心位置附近添加面积标签
        ax.text(x_center, y_center, f"{ann['area']:.1f}", color='white', fontsize=12, ha='center', va='center')
    ax.imshow(img)
    ax.imshow(img)
    plt.axis('off')
    
    # 将图像保存到内存中，并转换为NumPy数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_with_masks = np.array(Image.open(buf))
    
    plt.close()
    return img_with_masks


def create_roi_mask(frame_shape, x1, y1, x2, y2, label):
    """
    在指定的 ROI 区域内绘制一个蓝色框，并在框上方显示标签内容。

    参数：
    frame_shape (tuple): 原始图像的形状。
    x1, y1, x2, y2 (int): ROI区域的边界框坐标。
    label (str): 要显示的标签内容。

    返回：
    np.ndarray: 带有蓝色框和标签的mask数组。
    """
    height, width, _ = frame_shape
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    color = (0, 0, 255)
    thickness = 2
    cv2.rectangle(mask, (x1, y1), (x2, y2), color, thickness)
    
    # 在框上方显示标签内容
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10 + text_size[1]
    if label:
        cv2.putText(mask, label, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    return mask


def filter_masks(masks, roi):
    """
    过滤掉不符合特定条件的遮罩（masks）。

    参数：
    masks (list): 包含遮罩信息的字典列表，每个字典包含 'bbox' 键和相应的边界框信息。
    roi (tuple): 感兴趣区域的边界框，以 (x1, y1, x2, y2) 格式表示。

    返回：
    list: 过滤后的遮罩列表。
    """
    x1_roi, y1_roi, x2_roi, y2_roi = roi
    width_roi = x2_roi - x1_roi
    height_roi = y2_roi - y1_roi
    
    filtered_masks = []
    for mask in masks:
        x, y, w, h = mask['bbox']
        area = mask['area']
        if x == 0 or y == 0 or w == height_roi or h == width_roi:
            continue
        x += x1_roi
        y += y1_roi
        # 检查边界框是否在 ROI 的左半部分或右半部分之外
        if x > x1_roi + width_roi / 2:
            continue
        if x + w < x2_roi - width_roi / 2:
            continue
        # 检查边界框是否包含了 ROI 边框部分
        if x < x1_roi or x + w > x2_roi or y < y1_roi or y + h > y2_roi:
            continue
        # 检查面积是否小于 200 像素
        if area < 300 or area > 1500:
            continue
        filtered_masks.append(mask)
    
    return filtered_masks


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=False)
                video = gr.Video(label="Video", visible=True)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Video",
                    label="Input Type",
                )
                yolo_model_id = gr.Dropdown(
                    label="YOLO Model",
                    choices=[
                        "v10_remark/best.pt",
                        "v10x_finetune/best.pt",
                        "puncture_init/best.pt",
                    ],
                    value="v10_remark/best.pt",
                )
                segment_model_id = gr.Dropdown(
                    label="Segment Model",
                    choices=[
                        "vit_h",
                        "vit_l",
                        "vit_b",
                    ],
                    value="vit_l",
                )
                classify_model_id = gr.Dropdown(
                    label="Classify Model",
                    choices=[
                        "EfficientNet_23",
                    ],
                    value="EfficientNet_23"
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                yolo_conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.35,
                )
                classify_conf_threshold = gr.Slider(
                    label="Classify Confidence Threshold",
                    minimum=0.7,
                    maximum=1,
                    step=0.05,
                    value=0.9,
                )
                yolov10_infer = gr.Button(value="Detect Objects")
            
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=False)
                output_video = gr.Video(label="Annotated Video", visible=True)
        
        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            
            return image, video, output_image, output_video
        
        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )
        
        def run_inference(image, video,
                          yolo_model_id,
                          segment_model_id,
                          classify_model_id,
                          image_size,
                          yolo_conf_threshold,
                          classify_conf_threshold,
                          input_type):
            if input_type == "Image":
                return yolov10_inference(image, None,
                                         yolo_model_id,
                                         segment_model_id,
                                         classify_model_id,
                                         image_size,
                                         yolo_conf_threshold=yolo_conf_threshold,
                                         classify_conf_threshold=classify_conf_threshold)
            else:
                return yolov10_inference(None, video,
                                         yolo_model_id,
                                         segment_model_id,
                                         classify_model_id,
                                         image_size,
                                         yolo_conf_threshold=yolo_conf_threshold,
                                         classify_conf_threshold=classify_conf_threshold)
        
        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video,
                    yolo_model_id,
                    segment_model_id,
                    classify_model_id,
                    image_size,
                    yolo_conf_threshold,
                    classify_conf_threshold,
                    input_type],
            outputs=[output_image, output_video],
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Puncture Detection
    </h1>
    """)
    # gr.HTML(
    #     """
    #     <h3 style='text-align: center'>
    #     <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
    #     </h3>
    #     """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()
