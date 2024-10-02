import cv2
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from yolo_seg.utils.config import get_config
from yolo_seg.utils.needle_clasify import load_efficient_net, predict_and_find_start_inserted
from yolo_seg.utils.mask_tools import get_coord_min_rect_len
from yolo_seg.utils.speed_tools import plot_speeds, compute_metrics, gaussian_smoothing

CONFIG = get_config()

INIT_SHAFT_LEN = 20  # 针梗的实际长度，单位为毫米
MOVE_THRESHOLD = 2  # 针梗移动的阈值，单位为毫米
CONFIRMATION_FRAMES = 5  # 连续几帧确认像素比例和插入状态
OUT_EXPAND = 50  # 输出图像感兴趣区域的扩展像素数

video_info_dict = {}
deviations = {}


def process_video(video_path, yolo_model_id, classify_model_id, yolo_conf_threshold,
                  judge_wnd):
    print(f"Processing video: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
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
    lens = []
    last_rect_len = 0
    
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
        batch_size=yolo_batch_size)
    
    smooth_lens = gaussian_smoothing(lens)
    for idx, (rect_len, cls) in enumerate(zip(smooth_lens, class_list)):
        if cls == 0 and not inserted:
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
            print(insert_start_frame, insert_spec_end_frame, fps)
            interval_time = max(1, insert_spec_end_frame - insert_start_frame) / fps
            spec_insert_speed = MOVE_THRESHOLD / interval_time
    
    cap.release()
    video_info_dict[video_name] = {
        "start_frame": insert_start_frame,
        "end_frame": insert_spec_end_frame,
        "speed": spec_insert_speed,
    }
    
    # 生成速度折线图
    plt.figure()
    match = re.search(r'\d+', video_name)
    num = int(match.group())
    chart_path = f"speeds_chart/{video_name}.png"
    os.makedirs("../resources/speeds_chart", exist_ok=True)
    plot_speeds(lens, (insert_start_frame, insert_spec_end_frame), [actual_start, actual_end], chart_path)
    deviations[video_name] = compute_metrics(
        lens,
        (insert_start_frame, insert_spec_end_frame),
        fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    video_dir = os.path.join(CONFIG.PATH.DATASETS_PATH, "needle-seg/videos")
    parser.add_argument("-p", "--path", type=str, default=video_dir,
                        help="Path to video directory or file")
    parser.add_argument("-ym", "--yolo_model", type=str, default="seg/best.pt",
                        help="Path to YOLO model, e.g. seg/best.pt")
    parser.add_argument("-cm", "--classify_model", type=str, default="EfficientNet/EfficientNet_23.pkl",
                        help="Path to classification model, e.g. EfficientNet/EfficientNet_23.pkl")
    parser.add_argument("-yct", "--yolo_conf_threshold", type=float, default=0.35,
                        help="YOLO confidence threshold, default is 0.35")
    parser.add_argument("-jw", "--judge_wnd", type=int, default=20,
                        help="Window size for judging inserted needle, default is 20")
    parser.add_argument("-i", "--frame_interval", type=int, default=1,
                        help="Frame interval for calculating speed, default is 1")
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        for video in os.listdir(args.path):
            if video.endswith(".mp4"):
                video_path = os.path.join(args.path, video)
                process_video(video_path, args.yolo_model, args.classify_model,
                              args.yolo_conf_threshold, args.judge_wnd)
    else:
        process_video(args.path, args.yolo_model, args.classify_model,
                      args.yolo_conf_threshold, args.judge_wnd)
    for video, info in video_info_dict.items():
        print(f"{video}:  {info['start_frame']}-{info['end_frame']}  {info['speed']:.2f}mm/s")
    
    for video, deviation in deviations.items():
        print(f"{video} - Gaussian: {deviation[1]:.2f}, Normal: {deviation[0]:.2f}, "
              f"Savitzky Golay: {deviation[2]:.2f}")
    
    averages = [sum(values) / len(deviations) for values in zip(*deviations.values())]
    print(f"Avg - Gaussian: {averages[1]:.2f}, Normal: {averages[0]:.2f}, "
          f"Savitzky Golay: {averages[2]:.2f}")
