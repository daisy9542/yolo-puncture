import os
import cv2
import numpy as np
from tqdm import tqdm


def convert_davis_to_yolo(davis_images_dir, davis_annotations_dir, output_dir, class_map):
    """
    Convert DAVIS 2016 dataset annotations (masks) to YOLO format.

    davis_images_dir: Path to DAVIS images folder (JPEGImages).
    davis_annotations_dir: Path to DAVIS annotations folder (Annotations).
    output_dir: Path to save YOLO annotations.
    class_map: Dictionary mapping object class names to class IDs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each video in DAVIS dataset
    videos = os.listdir(davis_images_dir)
    for video in tqdm(videos):
        video_images_dir = os.path.join(davis_images_dir, video)
        video_annotations_dir = os.path.join(davis_annotations_dir, video)
        
        # Check if the video has corresponding annotation directory
        if not os.path.exists(video_annotations_dir):
            continue
        
        # Iterate through each frame in the video
        frames = os.listdir(video_images_dir)
        for frame_name in frames:
            # Read the image and mask
            image_path = os.path.join(video_images_dir, frame_name)
            mask_path = os.path.join(video_annotations_dir, frame_name.replace('.jpg', '.png'))
            
            if not os.path.exists(mask_path):
                continue
            
            # Load the image and mask
            img = cv2.imread(image_path)
            img_height, img_width, _ = img.shape
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 8-bit grayscale mask
            
            # Get contours (instances) from mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Prepare YOLO annotation file for the current frame
            yolo_annotation_file = os.path.join(output_dir, f'{video}_{frame_name[:-4]}.txt')
            with open(yolo_annotation_file, 'w') as annotation_file:
                # Process each contour (instance) in the mask
                for contour in contours:
                    if len(contour) < 5:
                        continue  # Ignore too small contours
                    
                    # Get bounding box from contour
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Normalize bounding box coordinates
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    
                    # Write YOLO format annotation (class_id x_center y_center width height)
                    # Assuming a single class for now, map class to ID
                    class_id = 0  # Replace with your class mapping logic
                    annotation_file.write(f'{class_id} {x_center} {y_center} {width} {height}\n')


# Example usage
davis_images_dir = '/mnt/zwp/DAVIS/JPEGImages/480p'
davis_annotations_dir = '/mnt/zwp/DAVIS/Annotations/480p'
output_dir = '/mnt/zwp/DAVIS/yolo_annotations'
class_map = {'object_class': 0}  # Example class mapping

convert_davis_to_yolo(davis_images_dir, davis_annotations_dir, output_dir, class_map)
