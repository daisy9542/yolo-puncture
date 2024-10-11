import os
import cv2
import re

from ultralytics import YOLO


def gen(model, video_path, output_directory, mode='train', key_index=0):

    file_name = os.path.basename(video_path)
    # 使用正则表达式提取 'video' 后面的数字
    match = re.search(r'video(\d+)\.mp4', file_name)
    video_no = match.group(1)    
    
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"无法打开视频文件 {video_path}")
        return
    
    frame_count = 0
    success, r_frame = video_capture.read()
    frame_list = []
    while success:
        frame_list.append(r_frame)
        success, r_frame = video_capture.read()
        frame_count +=1
    # 释放视频捕获对象
    video_capture.release()

    path1 = os.path.join(output_directory, 'images', mode)
    path2 = os.path.join(output_directory, 'labels', mode)
    os.makedirs(path1, exist_ok=True)
    os.makedirs(path2, exist_ok=True)

    for i,frame in enumerate(frame_list):
        # 构造帧的文件名
        image_name = f"{video_no}frame_{i}"
        image_path = os.path.join(path1, f"{image_name}.jpg")
        # 保存帧为图像文件
        cv2.imwrite(image_path, frame)

        # ------------------------------ 预测bbox ------------------------------
        result = model.predict(image_path, retina_masks=True)[0]
        boxes = result.boxes
        if len(boxes.cls) == 0:
            continue
        x,y,w,h = [val.item() for val in boxes.xywhn[0]]

        # print(boxes)
        with open(f"{path2}/{image_name}.txt", "w") as f:
                f.writelines(f"{int(i>=key_index)} {x} {y} {w} {h}")
        # ------------------------------ 预测bbox ------------------------------
    
    print(f"视频 {video_path} 处理完成，提取了 {frame_count} 帧。")
    

if __name__ == '__main__':

    
    # 定义模型
    model = YOLO("/home/puncture/weights/seg/yolo11n-seg-finetune.pt")
    out_dir = '/home/tsw/workspace/keyan/medical-puncture-tools/datasets/needle/'
    # 抽取帧和生成cls,bbox标签
    train_video_paths = [
        f'/home/puncture/datasets/needle-seg/videos/video{i}.mp4' for i in range(1,18)
    ]
    train_key_frames = [
        "1frame_122.jpg",
        "2frame_44.jpg",
        "3frame_1.jpg",
        "4frame_73.jpg",
        "5frame_27.jpg",
        "6frame_25.jpg",
        "7frame_14.jpg",
        "8frame_141.jpg",
        "9frame_12.jpg",
        "10frame_7.jpg",
        "11frame_3.jpg",
        "12frame_9.jpg",
        "13frame_9.jpg",
        "14frame_29.jpg",
        "15frame_30.jpg",
        "16frame_62.jpg",
        "17frame_151.jpg",
    ]

    val_video_paths = [
         '/home/puncture/datasets/needle-seg/videos/video18.mp4',
         '/home/puncture/datasets/needle-seg/videos/video19.mp4',
         '/home/puncture/datasets/needle-seg/videos/video27.mp4',
         '/home/puncture/datasets/needle-seg/videos/video28.mp4',
         '/home/puncture/datasets/needle-seg/videos/video29.mp4',
         '/home/puncture/datasets/needle-seg/videos/video30.mp4',
         '/home/puncture/datasets/needle-seg/videos/video31.mp4',
         '/home/puncture/datasets/needle-seg/videos/video32.mp4',
    ]
    val_key_frames = [
        "18frame_111.jpg",
        "19frame_60.jpg",
        "27frame_105.jpg",
        "28frame_74.jpg",
        "29frame_80.jpg",
        "30frame_85.jpg",
        "31frame_68.jpg",
        "32frame_34.jpg"
    ]
    for i,vpath in enumerate(train_video_paths):
        key_index = re.search(r"_(\d+)\.jpg", train_key_frames[i]).group(1)
        # mode= val 或者 train
        gen(model, vpath, out_dir,  mode='train', key_index=int(key_index))

