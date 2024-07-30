import gradio as gr
import cv2
import tempfile
import torch

from ultralytics import YOLOv10
import numpy as np
from utils.config import get_config
from utils.segment_everything import show_anns, segment
from utils.needle_clasify import load_efficient_net, predict_and_find_start_inserted
from utils.mask_tools import draw_masks_on_image, create_roi_mask, filter_masks, get_min_rect_len

CONFIG = get_config()
# PATH.WEIGHTS_PATH

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
                      judge_wnd):
    if yolo_model_id.endswith("pt"):
        model = YOLOv10(f'{yolo_model_id}')
    else:
        model = YOLOv10.from_pretrained(f'{yolo_model_id}/', local_files_only=True)
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=yolo_conf_threshold)
        annotated_image = results[0].plot()
        
        image_darray = np.array(image)
        anns = segment(image_darray, segment_model_id)
        np.save('masks.npy', anns)
        ann = filter_masks(anns)
        print(ann)
        if ann is None:
            img_with_masks = draw_masks_on_image(image, anns)
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
        insert_spec_counter = 0  # 判断插入皮肤超出长度计数器
        inserted = False  # 是否插入皮肤（只判断初始固定距离）
        # insert_start_time, insert_spec_end_time = None, None  # 记录插入皮肤的开始和指定结束时间
        insert_start_frame, insert_spec_end_frame = None, None  # 记录插入皮肤的开始和指定结束所在帧
        spec_insert_speed = None  # 插入皮肤指定长度的速度
        speed_clac_compute = False
        
        frames = []  # 帧列表
        yolo_batch_size = 4
        yolo_pred_results = []  # yolo的预测结果
        yolo_pred_boxes = []  # yolo预测的目标位置信息
        cls_model = load_efficient_net(name=classify_model_id)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        # yolo检测图片
        for i in range(0, len(frames), yolo_batch_size):
            batch_frames = frames[i:i + yolo_batch_size]
            pred_results = model.predict(source=batch_frames, imgsz=image_size, conf=yolo_conf_threshold)
            for pred_result in pred_results:
                yolo_pred_results.append(pred_result)
                yolo_pred_boxes.append(pred_result.boxes)
        
        class_list, prob_list, insert_start_frame = predict_and_find_start_inserted(cls_model,
                                                                                    frames=frames,
                                                                                    boxes_list=yolo_pred_boxes,
                                                                                    judge_wnd=judge_wnd,
                                                                                    batch_size=yolo_batch_size)
        # debug for: 查看分类结果
        s_cnt = 0
        for index, prob in zip(class_list, prob_list):
            if s_cnt>10 and s_cnt<230:
                print(f"{s_cnt},类: {index} => 概率: {round(prob * 100, 2)}%")
            s_cnt+=1
        
        for idx, (frame, pred_result, cls, prob) in enumerate(zip(frames, yolo_pred_results, class_list, prob_list)):
            pred_boxes = pred_result.boxes
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = np.zeros_like(frame)
            roi_mask = np.zeros_like(frame)
            
            # 若检测到多个物体，取置信度最大的
            if len(pred_boxes) == 0:
                continue
            best_conf_idx = torch.argmax(pred_boxes.conf)
            name = cls
            xyxy = pred_boxes.xyxy[best_conf_idx].squeeze()
            x1, y1, x2, y2 = map(int, xyxy.cpu().numpy())
            height, width, _ = frame.shape
            x1 = max(0, x1 - OUT_EXPAND)
            y1 = max(0, y1 - OUT_EXPAND)
            x2 = min(width, x2 + OUT_EXPAND)
            y2 = min(height, y2 + OUT_EXPAND)
            shaft_pixel_len = (x2 - x1) * (y2 - y1)  # 针梗的像素长度
            min_rect_len = shaft_pixel_len
            
            # 针梗分割
            if not speed_clac_compute and idx >= insert_start_frame - CONFIRMATION_FRAMES:
                roi = rgb_frame[y1:y2, x1:x2]
                anns = segment(roi, segment_model_id)
                ann = filter_masks(anns)
                # TODO: 可能找不到针梗
                if ann is None:
                    continue
                mask = show_anns(frame.shape, ann, x1, y1)
                _, _, w, h = ann['bbox']
                shaft_pixel_len = np.sqrt(w**2 + h**2)
                min_rect_len, _, _ = get_min_rect_len(ann)
            
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
            if idx == insert_start_frame:
                # insert_start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                inserted = True
            
            # 判断是否插入皮肤达到指定长度
            if cls == 1 and inserted and actual_len <= INIT_SHAFT_LEN - MOVE_THRESHOLD:
                # insert_spec_counter += 1
                # if insert_spec_counter >= CONFIRMATION_FRAMES:
                # insert_spec_counter = 0
                inserted = False
                speed_clac_compute = True
                insert_spec_end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                # insert_spec_end_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                # spec_insert_speed = 1000 * MOVE_THRESHOLD / (insert_spec_end_time - insert_start_time)
                interval_time = (insert_spec_end_frame - insert_start_frame) / fps
                spec_insert_speed = 1000 * MOVE_THRESHOLD / interval_time
            
            if speed_clac_compute:
                label = f"{name} {prob:.2f} {spec_insert_speed:.2f}mm/s"
            else:
                label = f"{name} {prob:.2f} {actual_len:.2f} {shaft_pixel_len:.2f}"
            
            roi_mask = create_roi_mask(frame.shape, x1, y1, x2, y2, label)
            # annotated_frame = pred_result.plot(label_content=label, best_conf_idx=best_conf_idx)
            combined_frame = cv2.addWeighted(frame, 1, ann, 1, 0)
            combined_frame = cv2.addWeighted(combined_frame, 1, roi_mask, 1, 0)
            out.write(combined_frame)
        
        cap.release()
        out.release()
        print("Start: ", insert_start_frame, " End: ", insert_spec_end_frame)
        
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
                        "EfficientNet/EfficientNet_23.pkl",
                    ],
                    value="EfficientNet/EfficientNet_23.pkl"
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
                judge_wnd = gr.Slider(
                    label="Window Size for Judging Insert-starting Frame",
                    minimum=10,
                    maximum=40,
                    step=5,
                    value=20,
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
                          judge_wnd,
                          input_type):
            if input_type == "Image":
                return yolov10_inference(image, None,
                                         yolo_model_id,
                                         segment_model_id,
                                         classify_model_id,
                                         image_size,
                                         yolo_conf_threshold=yolo_conf_threshold,
                                         judge_wnd=judge_wnd)
            else:
                return yolov10_inference(None, video,
                                         yolo_model_id,
                                         segment_model_id,
                                         classify_model_id,
                                         image_size,
                                         yolo_conf_threshold=yolo_conf_threshold,
                                         judge_wnd=judge_wnd)
        
        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video,
                    yolo_model_id,
                    segment_model_id,
                    classify_model_id,
                    image_size,
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
