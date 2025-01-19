""" 在图片的左上角标记图片的名字 """
import argparse
import os
import cv2


def add_text_to_image(input_path, output_path, *,
                      font_scale=2, color=(0, 0, 255),
                      thickness=2, padding_right=25, padding_top=60
                      ):
    image = cv2.imread(input_path)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    color = (0, 0, 204)
    thickness = 2
    text = os.path.basename(output_path).split(".")[0]
    # 获取文字的尺寸
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    image_height, image_width = image.shape[:2]
    padding_top = 20  # 距离顶部的像素数
    padding_right = 20  # 距离右边的像素数
    text_x = image_width - text_size[0] - padding_right
    text_y = padding_top + text_size[1]  # OpenCV 的 y 坐标是文字的基线
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在图片右上角添加红色文字")
    parser.add_argument("input_path", type=str, help="输入图片路径")
    parser.add_argument("output_path", type=str, help="输出图片目录")
    
    args = parser.parse_args()
    add_text_to_image(args.input_path, args.output_path)
