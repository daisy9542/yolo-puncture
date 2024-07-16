import os
import re

labels_dir = '/home/puncture/datasets/needle/labels'

video_numbers = set()
pattern = re.compile(r'(\d+)frame_\d+\.txt')

for root, dirs, files in os.walk(labels_dir):
    for file in files:
        if file.endswith('.txt'):
            match = pattern.match(file)
            if match:
                video_number = match.group(1)
                video_numbers.add(int(video_number))

video_numbers = sorted(video_numbers)
key_frame_dir = {i: -1 for i in video_numbers}
sub_dirs = ["train", "val", "test"]

for video_number in video_numbers:
    frame = 0
    while True:
        label_file = f"{video_number}frame_{frame}.txt"
        label_path = ""
        for sub_dir in sub_dirs:
            label_path = os.path.join(labels_dir, sub_dir, label_file)
            if os.path.exists(label_path):
                break
        with open(label_path, 'r') as file:
            lines = file.readlines()
            parts = lines[0].strip().split()
            if len(parts) == 5 and parts[0] == '1':
                key_frame_dir[video_number] = frame
                break
        frame += 1

for k, v in key_frame_dir.items():
    print(f"Video{k}: {v}")
