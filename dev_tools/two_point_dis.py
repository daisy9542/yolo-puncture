import cv2
import numpy as np
import os

# 视频路径
video_path = "../datasets/videos/video10.mp4"
# 保存图片的文件夹
image_folder = f"extracted_frames/video10/"
os.makedirs(image_folder, exist_ok=True)

# 提取视频帧并保存为图片
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {frame_count}, FPS: {fps}")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_list = []
for i in range(frame_count):
    ret, frame = cap.read()
    if ret:
        frame_name = os.path.join(image_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_name, frame)
        frame_list.append(frame_name)
    else:
        break
cap.release()

# 全局变量，用于存储点击的点和文件路径
points = []
img_path = ""
current_index = 0


def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("Image", img)
        if len(points) == 2:
            pt1 = np.array(points[0])
            pt2 = np.array(points[1])
            distance = np.linalg.norm(pt1 - pt2)
            print(f"Distance: {distance:.2f} pixels")
            mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(img, f"{distance:.2f}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow("Image", img)


def process_next_image():
    global img, img_path, points, current_index
    if current_index < len(frame_list):
        img_path = frame_list[current_index]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image {img_path}")
            current_index += 1
            process_next_image()
        else:
            points = []
            display_image_with_progress()
            cv2.setMouseCallback("Image", click_event)
    else:
        print("All images processed.")
        cv2.destroyAllWindows()


def display_image_with_progress():
    global current_index
    img_with_text = img.copy()
    img_name = os.path.basename(img_path)
    progress_text = f"{img_name} {current_index + 1}/{len(frame_list)}"
    cv2.putText(img_with_text, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", img_with_text)


def on_key(event, x, y, flags, param):
    global current_index
    if event == cv2.EVENT_KEYDOWN:
        if x == ord('\r'):  # Enter key
            current_index += 1
            process_next_image()


# 设置初始窗口
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_event)
cv2.setMouseCallback("Image", on_key)

if frame_list:
    process_next_image()
    while True:
        key = cv2.waitKey(0)
        if key == 13:  # Enter key
            current_index += 1
            process_next_image()
        elif key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
else:
    print("No frames extracted from the video.")
