
import os
import cv2
import re



def extract_frames(video_path, output_directory='/home/puncture/datasets/needle-seg/extract_tmp'):
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

    tmp_path = os.path.join(output_directory, video_no)
    os.makedirs(tmp_path, exist_ok=True)

    for i,frame in enumerate(frame_list):
        # 构造帧的文件名
        image_name = f"{video_no}frame_{i}"
        
        image_path = os.path.join(tmp_path, f"{image_name}.jpg")
        # 保存帧为图像文件
        cv2.imwrite(image_path, frame)

    print(f"视频 {video_no} 处理完成，提取了 {frame_count} 帧。")

if __name__ == '__main__':
    vpath_list = [
        f'/home/puncture/datasets/needle-seg/videos/video{i}.mp4' for i in range(28,30)
    ]
    for vpath in vpath_list:
        extract_frames(vpath)

    """
    最终观察标记处关键插入帧：

    """