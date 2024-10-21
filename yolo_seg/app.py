import gradio as gr
import cv2
import tempfile
import numpy as np
import torch.cuda

from ultralytics import YOLO
from tasks import (
    load_efficient_net,
    predict_and_find_start_inserted,
)
from utils import (
    get_config,
    get_coord_mask,
    create_roi_mask,
    get_coord_min_rect_len,
    crop_frame,
    gaussian_smoothing,
)

CONFIG = get_config()

INIT_SHAFT_LEN = 20  # 针梗的实际长度，单位为毫米
MOVE_THRESHOLD = 2  # 针梗移动的阈值，单位为毫米
CONFIRMATION_FRAMES = 5  # 连续几帧确认像素比例和插入状态
OUT_EXPAND = 50  # 输出图像感兴趣区域的扩展像素数

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def yolo_inference(image, video,
                   yolo_model_id,
                   classify_model_id,
                   yolo_conf_threshold,
                   judge_wnd
                   ):
    model = YOLO(f'{CONFIG.PATH.WEIGHTS_PATH}/{yolo_model_id}')
    if image:
        results = model.predict(source=image, conf=yolo_conf_threshold, retina_masks=True, device=device)
        seg_coords = results[0].masks.xy[0]
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = get_coord_mask(image.shape, seg_coords)
        annotated_image = cv2.addWeighted(image, 1, mask, 1, 0)
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
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'avc1'), fps, (frame_width, frame_height))
        
        yolo_pred_xyxy = []  # yolo 预测的目标位置信息
        coord_xys = []  # 实例分割标注数组
        last_box = None  # 上一帧的目标位置信息
        frames = []  # 帧列表
        yolo_batch_size = 4
        pixel_len_arr = []  # 视频中针梗的长度，以像素为单位
        inserted = False  # 是否插入皮肤（只判断初始固定距离）
        insert_start_frame, insert_spec_end_frame = None, None  # 记录插入皮肤的开始和指定结束所在帧
        spec_insert_speed = None  # 插入皮肤指定长度的速度
        speed_clac_compute = False  # 是否开始计算插入皮肤的速度
        lens = []  # 存储像素长度
        last_rect_len = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
            results = model.predict(source=frame, conf=yolo_conf_threshold, retina_masks=True, device=device)
            pred_boxes = results[0].boxes.cpu().numpy()
            height, width, _ = frame.shape
            
            if len(pred_boxes.cls) > 0:
                # 若检测到多个物体，取置信度最大的
                best_conf_idx = np.argmax(pred_boxes.conf)
                xyxy_box = pred_boxes.xyxy[best_conf_idx].squeeze()
                xyxy_box = list(map(int, xyxy_box))
                last_box = xyxy_box
                seg_mask = results[0].masks.xy[best_conf_idx]
                coord_xys.append(seg_mask)
                rect_len, _ = get_coord_min_rect_len(seg_mask)
                last_rect_len = rect_len
                lens.append(rect_len)
            else:
                if last_box is None:
                    xyxy_box = 0, 0, width, height
                else:
                    xyxy_box = last_box
                coord_xys.append(None)
                lens.append(last_rect_len)
            
            yolo_pred_xyxy.append(xyxy_box)
        
        cls_model = load_efficient_net(name=classify_model_id)
        class_list, prob_list, insert_start_frame = predict_and_find_start_inserted(
            cls_model,
            frames=frames,
            boxes_list=yolo_pred_xyxy,
            judge_wnd=judge_wnd,
            batch_size=yolo_batch_size
        )
        
        crop_frames = map(crop_frame, frames, yolo_pred_xyxy)
        
        last_xyxy = None
        smooth_lens = gaussian_smoothing(lens)
        for idx, (frame, coord_xy, rect_len, xyxy, cls, prob) in enumerate(
                zip(frames, coord_xys, smooth_lens, yolo_pred_xyxy, class_list, prob_list)
        ):
            height, width, _ = frame.shape
            
            if inserted:
                x1, y1, x2, y2 = last_xyxy
            else:
                x1, y1, x2, y2 = xyxy
                x1 = max(0, x1 - OUT_EXPAND)
                y1 = max(0, y1 - OUT_EXPAND)
                x2 = min(width, x2 + OUT_EXPAND)
                y2 = min(height, y2 + OUT_EXPAND)
                last_xyxy = x1, y1, x2, y2
            
            if cls == 0 and not inserted and coord_xy is not None:
                pixel_len_arr.append(rect_len)
                if len(pixel_len_arr) > CONFIRMATION_FRAMES:
                    pixel_len_arr.pop(0)
            if cls == 1 and len(pixel_len_arr) == 0:
                # 第一帧就检测到插入皮肤的情况
                if rect_len is None:
                    continue
                else:
                    pixel_len_arr.append(rect_len)
            actual_len = INIT_SHAFT_LEN if cls == 0 else (
                    INIT_SHAFT_LEN * rect_len / (sum(pixel_len_arr) / len(pixel_len_arr)))
            
            # 判断是否开始插入皮肤
            if idx == insert_start_frame:
                inserted = True
            
            # 判断是否插入皮肤达到指定长度
            if cls == 1 and inserted and actual_len <= INIT_SHAFT_LEN - MOVE_THRESHOLD:
                inserted = False
                speed_clac_compute = True
                insert_spec_end_frame = idx
                interval_time = max(1, insert_spec_end_frame - insert_start_frame) / fps
                spec_insert_speed = MOVE_THRESHOLD / interval_time
            
            if speed_clac_compute:
                label = f"{idx} {cls} {prob:.2f} {spec_insert_speed:.2f}mm/s"
            elif rect_len is None:
                label = f"{idx} {cls} {prob:.2f} {actual_len:.2f} -"
            else:
                label = f"{idx} {cls} {prob:.2f} {actual_len:.2f} {rect_len:.2f}"
            mask = get_coord_mask(frame.shape, coord_xy)
            roi_mask = create_roi_mask(frame.shape, x1, y1, x2, y2, label)
            combined_frame = cv2.addWeighted(frame, 1, mask, 1, 0)
            combined_frame = cv2.addWeighted(combined_frame, 1, roi_mask, 1, 0)
            out.write(combined_frame)
        cap.release()
        out.release()
        print(f"Start: {insert_start_frame} End: {insert_spec_end_frame} Speed: {spec_insert_speed:.2f}mm/s")
        
        # # 生成速度折线图
        # plt.figure()
        #
        # chart_path = tempfile.mktemp(suffix=".png")
        # plot_speeds(lens, (insert_start_frame, insert_spec_end_frame), file_path=chart_path)
        #
        return None, output_video_path


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
                        "seg/yolov8n-seg-finetune.pt",
                        "seg/yolo11n-seg-finetune.pt",
                        "seg/yolo11x-seg-finetune.pt",
                    ],
                    value="seg/yolo11n-seg-finetune.pt",
                )
                classify_model_id = gr.Dropdown(
                    label="Classify Model",
                    choices=[
                        "EfficientNet/EfficientNet_23.pkl",
                    ],
                    value="EfficientNet/EfficientNet_23.pkl"
                )
                yolo_conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.35,
                )
                judge_wnd = gr.Slider(
                    label="Window Size for Judging Insert-starting Frame",
                    minimum=10,
                    maximum=40,
                    step=5,
                    value=20,
                )
                yolo_infer = gr.Button(value="Detect Objects")
            
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
                          classify_model_id,
                          yolo_conf_threshold,
                          judge_wnd,
                          input_type
                          ):
            if input_type == "Image":
                return yolo_inference(image, None,
                                      yolo_model_id,
                                      classify_model_id,
                                      yolo_conf_threshold=yolo_conf_threshold,
                                      judge_wnd=judge_wnd
                                      )
            else:
                return yolo_inference(None, video,
                                      yolo_model_id,
                                      classify_model_id,
                                      yolo_conf_threshold=yolo_conf_threshold,
                                      judge_wnd=judge_wnd
                                      )
        
        yolo_infer.click(
            fn=run_inference,
            inputs=[image, video,
                    yolo_model_id,
                    classify_model_id,
                    yolo_conf_threshold,
                    judge_wnd,
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
    """
    )
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch(ssl_verify=False)
