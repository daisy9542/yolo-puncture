"""
自动计算指定视频或者目录下所有视频的针梗穿刺速度，并计算误差（如果参考值可用）。
该文件需要与 `needle_seg/app.py` 同步。
"""
import cv2
import os
import argparse
import numpy as np
from natsort import natsorted
from needle_seg.tasks import (
    load_efficient_net,
    predict_and_find_start_inserted,
)
from needle_seg.utils import (
    get_config,
    get_coord_min_rect_len,
    crop_frame,
    compute_metrics,
    gaussian_smoothing,
)
from needle_seg.vis_prompt.workflow_cuite import process_video_with_yolo_and_cutie

CONFIG = get_config()

INIT_SHAFT_LEN = 20  # 针梗的实际长度，单位为毫米
MOVE_THRESHOLD = 2  # 针梗移动的阈值，单位为毫米
CONFIRMATION_FRAMES = 5  # 连续几帧确认像素比例和插入状态
OUT_EXPAND = 50  # 输出图像感兴趣区域的扩展像素数

video_info_dict = {}
deviations = {}


def process_video(video_path, yolo_model_path, classify_model_id, judge_wnd):
    print(f"Processing video: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    mask_list = process_video_with_yolo_and_cutie(yolo_model_path, video_path)
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
    frame_idx = 0  # 用于索引 mask_list 中的掩码
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        # 获取当前帧对应的二值掩码
        if frame_idx < len(mask_list):
            binary_mask = mask_list[frame_idx].cpu().numpy()  # 二值掩码
            frame_idx += 1
            
            # 查找二值掩码中的目标（前景像素）
            if np.any(binary_mask):  # 如果存在前景
                # 获取前景目标的边界框（xyxy格式）
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_contour = max(contours, key=cv2.contourArea)  # 找到面积最大的轮廓
                x, y, w, h = cv2.boundingRect(max_contour)
                xyxy_box = [x, y, x + w, y + h]
                last_box = xyxy_box
                
                # 获取掩码的坐标列表（非零点的坐标）
                seg_mask = np.column_stack(np.where(binary_mask > 0))
                coord_xys.append(seg_mask)
                
                # 计算掩码的最小外接矩形长度
                rect_len, _ = get_coord_min_rect_len(seg_mask)
                last_rect_len = rect_len
                lens.append(rect_len)
            else:
                # 如果当前帧没有前景目标
                if last_box is None:
                    x, y, w, h = 0, 0, frame.shape[1], frame.shape[0]
                    xyxy_box = [x, y, w, h]
                else:
                    xyxy_box = last_box
                coord_xys.append(None)
                lens.append(last_rect_len)
        else:
            # 如果没有更多掩码了，停止处理
            break
        
        yolo_pred_xyxy.append(xyxy_box)
    
    cap.release()
    
    cls_model = load_efficient_net(name=classify_model_id)
    class_list, prob_list, insert_start_frame = predict_and_find_start_inserted(
        cls_model,
        frames=frames,
        boxes_list=yolo_pred_xyxy,
        judge_wnd=judge_wnd,
        batch_size=yolo_batch_size)
    
    crop_result = map(crop_frame, frames, yolo_pred_xyxy)
    
    smooth_lens = gaussian_smoothing(lens)
    for idx, ((cropped_frame, cropped_coord), frame, coord_xy, rect_len, xyxy, cls, prob) in enumerate(
            zip(crop_result, frames, coord_xys, smooth_lens, yolo_pred_xyxy, class_list, prob_list)
    ):
        height, width, _ = frame.shape
        if cls == 0 and not inserted and coord_xy is not None:
            pixel_len_arr.append(rect_len)
            if len(pixel_len_arr) > CONFIRMATION_FRAMES:
                pixel_len_arr.pop(0)
        if cls == 1 and len(pixel_len_arr) == 0:
            if rect_len is None:
                continue
            else:
                pixel_len_arr.append(rect_len)
        if len(pixel_len_arr) == 0:
            continue
        actual_len = INIT_SHAFT_LEN if cls == 0 else (
                INIT_SHAFT_LEN * rect_len / (sum(pixel_len_arr) / len(pixel_len_arr)))
        
        if idx == insert_start_frame:
            inserted = True
        
        if cls == 1 and inserted and actual_len <= INIT_SHAFT_LEN - MOVE_THRESHOLD:
            inserted = False
            insert_spec_end_frame = idx
            interval_time = max(1, insert_spec_end_frame - insert_start_frame) / fps
            spec_insert_speed = MOVE_THRESHOLD / interval_time
    
    cap.release()
    video_info_dict[video_name] = {
        "start_frame": insert_start_frame,
        "end_frame": insert_spec_end_frame,
        "speed": spec_insert_speed,
    }
    
    # 生成速度折线图
    # plt.figure()
    # match = re.search(r'\d+', video_name)
    # chart_path = f"resources/speeds_chart/{video_name}.png"
    # os.makedirs("../resources/speeds_chart", exist_ok=True)
    # plot_speeds(lens, (insert_start_frame, insert_spec_end_frame), file_path=chart_path)
    # deviations[video_name] = compute_metrics(
    #     lens,
    #     (insert_start_frame, insert_spec_end_frame),
    #     (),
    #     fps=fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    video_dir = os.path.join(CONFIG.PATH.DATASETS_PATH, "videos")
    parser.add_argument("-p", "--path", type=str, default=video_dir,
                        help="Path to video directory or file")
    parser.add_argument("-ym", "--yolo_model", type=str, default="seg/yolo11n-seg-finetune.pt",
                        help="Path to YOLO model, e.g. seg/yolo11n-seg-finetune.pt")
    parser.add_argument("-cm", "--classify_model", type=str, default="EfficientNet/EfficientNet_23.pkl",
                        help="Path to classification model, e.g. EfficientNet/EfficientNet_23.pkl")
    parser.add_argument("-yct", "--yolo_conf_threshold", type=float, default=0.35,
                        help="YOLO confidence threshold, default is 0.35")
    parser.add_argument("-jw", "--judge_wnd", type=int, default=20,
                        help="Window size for judging inserted needle, default is 20")
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        for video in os.listdir(args.path):
            if video.endswith(".mp4"):
                video_path = os.path.join(args.path, video)
                process_video(video_path, args.yolo_model, args.classify_model, args.judge_wnd)
    else:
        process_video(args.path, args.yolo_model, args.classify_model, args.judge_wnd)
    for video in natsorted(video_info_dict.keys()):
        info = video_info_dict[video]
        print(f"{video}:  {info['start_frame']}-{info['end_frame']}  {info['speed']:.2f}mm/s")
    
    # for video, deviation in deviations.items():
    #     print(f"{video} - Gaussian: {deviation[1]:.2f}, Normal: {deviation[0]:.2f}, "
    #           f"Savitzky Golay: {deviation[2]:.2f}")
    #
    # averages = [sum(values) / len(deviations) for values in zip(*deviations.values())]
    # print(f"Avg - Gaussian: {averages[1]:.2f}, Normal: {averages[0]:.2f}, "
    #       f"Savitzky Golay: {averages[2]:.2f}")
