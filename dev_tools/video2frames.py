"""
将（目录下所有的）视频文件所有帧保存为图片。
"""
import os
import cv2
import argparse

def video2frames(path):
    if os.path.isdir(path):
        # Loop over files in the directory
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if os.path.isfile(filepath) and is_video_file(filename):
                process_video(filepath)
    elif os.path.isfile(path):
        if is_video_file(path):
            process_video(path)
        else:
            print(f"'{path}' is not a recognized video file.")
    else:
        print(f"'{path}' is not a valid file or directory.")

def is_video_file(filename):
    # List of common video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    extension = os.path.splitext(filename)[1].lower()
    return extension in video_extensions

def process_video(video_path):
    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Get the parent directory of the video file
    parent_dir = os.path.dirname(video_path)
    # Create the output directory
    output_dir = os.path.join(parent_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    while success:
        # Construct the frame filename
        video_name = video_name.replace("video", "")
        frame_filename = f"{video_name}frame_{frame_count}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        # Save the frame as axn image
        cv2.imwrite(frame_path, frame)
        # Read the next frame
        success, frame = cap.read()
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames from '{video_name}' to '{output_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from video files.')
    parser.add_argument('-p', '--path', required=True, help='Path to a video file or a directory containing video files.')
    args = parser.parse_args()
    video2frames(args.path)