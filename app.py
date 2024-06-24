import gradio as gr
import cv2
import tempfile
from ultralytics import YOLOv10
import numpy as np
import torch

INIT_STEM_LEN = 20


def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    # model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    if model_id.endswith("pt"):
        model = YOLOv10(f'./runs/detect/{model_id}')
    else:
        model = YOLOv10.from_pretrained(f'./weights/{model_id}/', local_files_only=True)
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
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
        
        prev_gray = None
        prev_frame = None
        
        pixel_len = -1  # 视频中针梗的长度，以像素为单位
        count = 0  # 计数器，用于计算针梗的平均实际长度
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 将帧转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(frame)
            
            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            pred_boxes = results[0].boxes
            
            if len(pred_boxes.cls) > 0:
                # 若检测到多个物体，取置信度最大的
                best_conf_idx = torch.argmax(pred_boxes.conf)
                cls, conf = int(pred_boxes.cls[best_conf_idx]), float(pred_boxes.conf[best_conf_idx])
                name = results[0].names[cls]
                xyxy = pred_boxes.xyxy[best_conf_idx].squeeze()
                x1, y1, x2, y2 = xyxy.cpu().numpy()
                degree = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                stem_len = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if cls == 0:
                    pixel_len = pixel_len * count + stem_len
                    count += 1
                    pixel_len /= count
                actual_len = INIT_STEM_LEN if cls == 0 else stem_len / pixel_len * INIT_STEM_LEN
                label = f"{name} {conf:.2f} {degree:.2f} {actual_len:.2f}mm"
                
                x1, y1, x2, y2 = map(int, xyxy.cpu().numpy())
                
                if prev_gray is not None:
                    # Farneback 光流法
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5,
                        3, 15, 3, 5, 1.2, 0)
                    mask = draw_dense_flow(frame, flow)
                
                prev_gray = gray.copy()
            else:
                label = None
                best_conf_idx = None
            annotated_frame = results[0].plot(label_content=label, best_conf_idx=best_conf_idx)
            combined_frame = cv2.addWeighted(annotated_frame, 1, mask, 1, 0)
            out.write(combined_frame)
        
        cap.release()
        out.release()
        
        return None, output_video_path


def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov10_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image


def draw_flow(img, flow, step=16):
    """在图像上绘制光流矢量箭头，可以直接展示物体的移动方向和幅度。
    这种方法通常用于对光流场的视觉分析。"""
    h, w = img.shape[:2]
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return vis


def draw_hsv(flow):
    """将光流矢量转换为色相（Hue）和强度（Intensity），然后用颜色编码显示，
    这种方法可以用来分析整个图像的运动情况。"""
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    ang = np.arctan2(fy, fx) + np.pi
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def draw_dense_flow(img, flow):
    """使用稠密光流箭头绘制每个像素的光流矢量，通过绘制箭头可以观察到局部运动的方向和速度。"""
    h, w = img.shape[:2]
    vis = np.zeros((h, w, 3), np.uint8)
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            fx, fy = flow[y, x]
            cv2.arrowedLine(vis, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.3)
    return vis


def draw_direction(flow):
    """将光流矢量转换为图像的亮度，并显示不同方向的光流，用于查看图像中不同区域的运动趋势。"""
    fx, fy = flow[..., 0], flow[..., 1]
    ang = np.arctan2(fy, fx) + np.pi
    magnitude, angle = cv2.cartToPolar(fx, fy)
    direction = np.uint8(ang * 180 / np.pi / 2)
    direction = cv2.applyColorMap(direction, cv2.COLORMAP_HSV)
    return direction


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Video",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "train16/weights/best.pt",
                        "train16/weights/last.pt",
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                    ],
                    value="train16/weights/best.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.35,
                )
                yolov10_infer = gr.Button(value="Detect Objects")
            
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)
        
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
        
        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov10_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov10_inference(None, video, model_id, image_size, conf_threshold)
        
        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )
        
        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov10s",
                    640,
                    0.25,
                ],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()
