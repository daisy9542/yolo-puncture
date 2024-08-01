import cv2
import os
import argparse
import numpy as np

from ultralytics import YOLO
from utils.config import get_config
from utils.needle_clasify import load_efficient_net, predict_and_find_start_inserted
from utils.mask_tools import get_coord_mask, create_roi_mask, get_coord_min_rect_len

CONFIG = get_config()

INIT_SHAFT_LEN = 20  # 针梗的实际长度，单位为毫米
MOVE_THRESHOLD = 2  # 针梗移动的阈值，单位为毫米
CONFIRMATION_FRAMES = 5  # 连续几帧确认像素比例和插入状态
OUT_EXPAND = 50  # 输出图像感兴趣区域的扩展像素数

video_info_dict = {}


def process_video(video_path, yolo_model_id, classify_model_id, yolo_conf_threshold,
                  judge_wnd):
    print(f"Processing video: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))
    model = YOLO(f'{yolo_model_id}')
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    yolo_pred_xyxy = []
    coord_xys = []
    last_box = None
    frames = []
    yolo_batch_size = 4
    pixel_len_arr = []
    inserted = False
    insert_start_frame, insert_spec_end_frame = None, None
    spec_insert_speed = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        results = model.predict(source=frame, conf=yolo_conf_threshold)
        pred_boxes = results[0].boxes.cpu().numpy()
        height, width, _ = frame.shape
        
        if len(pred_boxes.cls) > 0:
            best_conf_idx = np.argmax(pred_boxes.conf)
            xyxy_box = pred_boxes.xyxy[best_conf_idx].squeeze()
            xyxy_box = list(map(int, xyxy_box))
            last_box = xyxy_box
            seg_mask = results[0].masks.xy[best_conf_idx]
            coord_xys.append(seg_mask)
        else:
            if last_box is None:
                xyxy_box = 0, 0, width, height
            else:
                xyxy_box = last_box
            coord_xys.append(None)
        
        yolo_pred_xyxy.append(xyxy_box)
    
    cls_model = load_efficient_net(name=classify_model_id)
    class_list, prob_list, insert_start_frame = predict_and_find_start_inserted(
        cls_model,
        frames=frames,
        boxes_list=yolo_pred_xyxy,
        judge_wnd=judge_wnd,
        batch_size=yolo_batch_size)
    
    last_rect_len = None
    for idx, (coord_xy, cls) in enumerate(zip(coord_xys, class_list)):
        if coord_xy is not None:
            rect_len, _ = get_coord_min_rect_len(coord_xy)
            last_rect_len = rect_len
        else:
            rect_len = last_rect_len
        
        if cls == 0 and not inserted and coord_xy is not None:
            pixel_len_arr.append(rect_len)
            if len(pixel_len_arr) > CONFIRMATION_FRAMES:
                pixel_len_arr.pop(0)
        if cls == 1 and len(pixel_len_arr) == 0:
            if rect_len is None:
                continue
            else:
                pixel_len_arr.append(rect_len)
        actual_len = INIT_SHAFT_LEN if cls == 0 else (
                INIT_SHAFT_LEN * rect_len / (sum(pixel_len_arr) / len(pixel_len_arr)))
        
        if idx == insert_start_frame:
            inserted = True
        
        if cls == 1 and inserted and actual_len <= INIT_SHAFT_LEN - MOVE_THRESHOLD:
            inserted = False
            insert_spec_end_frame = idx
            interval_time = (insert_spec_end_frame - insert_start_frame) / fps
            spec_insert_speed = MOVE_THRESHOLD / interval_time
    
    cap.release()
    video_info_dict[video_name] = {
        "start_frame": insert_start_frame,
        "end_frame": insert_spec_end_frame,
        "speed": spec_insert_speed
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model", type=str, default="seg/best.pt")
    parser.add_argument("--classify_model", type=str, default="EfficientNet/EfficientNet_23.pkl")
    parser.add_argument("--yolo_conf_threshold", type=float, default=0.35)
    parser.add_argument("--judge_wnd", type=int, default=20)
    args = parser.parse_args()
    
    video_dir = os.path.join(CONFIG.PATH.DATASETS_PATH, "needle-seg/videos")
    for video in os.listdir(video_dir):
        if video.endswith(".mp4"):
            video_path = os.path.join(video_dir, video)
            process_video(video_path, args.yolo_model, args.classify_model,
                          args.yolo_conf_threshold, args.judge_wnd)
    for video, info in video_info_dict.items():
        print(f"{video}:  {info['start_frame']}-{info['end_frame']}  {info['speed']:.2f}mm/s")
