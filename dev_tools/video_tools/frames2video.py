import cv2
import os
from argparse import ArgumentParser

from yolo_seg.utils.video_reader import sort_key


def frames2video(image_folder, output_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images = sorted(images, key=sort_key)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter.fourcc(*'avc1')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: {img_path} couldn't be read")
            continue
        video.write(img)
    
    # 释放资源
    video.release()
    
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--output_path", type=str, required=False, help="Path to the output video")
    parser.add_argument("--fps", type=int, required=False, default=30, help="Frames per second")
    args = parser.parse_args()
    
    video_name = os.path.basename(args.image_folder)
    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(args.image_folder), f"{video_name}.mp4")
    
    frames2video(args.image_folder, args.output_path, args.fps)
