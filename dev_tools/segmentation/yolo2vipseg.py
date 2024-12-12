"""
将 YOLO-seg 的数据集转换为 VIPSeg 格式的数据集
"""
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import yaml
import argparse

def create_vipseg_dataset(source_dir, target_dir):
    image_dir = os.path.join(source_dir, "images")
    label_dir = os.path.join(source_dir, "labels")

    target_imgs_dir = os.path.join(target_dir, "imgs")
    target_panomasks_dir = os.path.join(target_dir, "panomasks")

    os.makedirs(target_imgs_dir, exist_ok=True)
    os.makedirs(target_panomasks_dir, exist_ok=True)

    train_list = []
    val_list = []
    test_list = []

    # 解析 data.yaml 文件
    yaml_path = os.path.join(source_dir, "data.yaml")
    with open(yaml_path, "r") as f:
        data_config = yaml.safe_load(f)
    names = data_config.get("names", {})

    # 遍历数据集划分
    for split in ["train", "val", "test"]:
        print(f"Processing {split} set...")
        split_image_dir = os.path.join(image_dir, split)
        split_label_dir = os.path.join(label_dir, split)
        if not os.path.exists(split_image_dir):
            print(f"Split {split} does not exist, skipping...")
            continue

        frame_counter = 1

        # 遍历每一帧
        for filename in tqdm(sorted(os.listdir(split_image_dir))):
            if filename.endswith(".jpg"):
                video_name = f'video{filename.split("frame_")[0]}'

                # 创建视频目录
                video_imgs_dir = os.path.join(target_imgs_dir, video_name)
                video_masks_dir = os.path.join(target_panomasks_dir, video_name)
                os.makedirs(video_imgs_dir, exist_ok=True)
                os.makedirs(video_masks_dir, exist_ok=True)

                # 生成四位数字文件名
                frame_name = f"{frame_counter:04d}"

                # 处理图像文件
                image_path = os.path.join(split_image_dir, filename)
                label_path = os.path.join(split_label_dir, filename.replace(".jpg", ".txt"))

                # 保存图像文件到目标目录
                target_image_path = os.path.join(video_imgs_dir, f"{frame_name}.jpg")
                os.system(f"cp {image_path} {target_image_path}")

                # 添加到分割列表
                if split == "train":
                    train_list.append(f"imgs/{video_name}/{frame_name}.jpg")
                elif split == "val":
                    val_list.append(f"imgs/{video_name}/{frame_name}.jpg")
                elif split == "test":
                    test_list.append(f"imgs/{video_name}/{frame_name}.jpg")

                # 读取图像尺寸
                image = cv2.imread(image_path)
                height, width, _ = image.shape

                # 创建灰度掩码
                mask = np.zeros((height, width), dtype=np.uint8)

                # 读取标签文件
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            class_id = int(parts[0])  # 类别ID
                            coordinates = list(map(float, parts[1:]))

                            # 解析多边形坐标
                            polygon = []
                            for i in range(0, len(coordinates), 2):
                                x = int(coordinates[i] * width)
                                y = int(coordinates[i + 1] * height)
                                polygon.append((x, y))

                            # 绘制到掩码图像上
                            cv2.fillPoly(mask, [np.array(polygon)], 255)

                # 保存掩码图像
                target_mask_path = os.path.join(video_masks_dir, f"{frame_name}.png")
                cv2.imwrite(target_mask_path, mask)

                frame_counter += 1

    # 写入 train.txt, val.txt, test.txt
    with open(os.path.join(target_dir, "train.txt"), "w") as f:
        f.writelines([line + "\n" for line in sorted(train_list)])

    with open(os.path.join(target_dir, "val.txt"), "w") as f:
        f.writelines([line + "\n" for line in sorted(val_list)])

    with open(os.path.join(target_dir, "test.txt"), "w") as f:
        f.writelines([line + "\n" for line in sorted(test_list)])

    # 生成类别文件
    label_dict = {str(k): v for k, v in names.items()}
    with open(os.path.join(target_dir, "label_num_dic_final.json"), "w") as f:
        json.dump(label_dict, f, indent=4)

    print("Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a yolo-seg dataset to VIPSeg format')
    parser.add_argument('source', type=str, help='Path to the source dataset directory')
    parser.add_argument('target', type=str, help='Path to the target VIPSeg dataset directory')
    args = parser.parse_args()
    create_vipseg_dataset(args.source, args.target)