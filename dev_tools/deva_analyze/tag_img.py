""" 在图片的左上角标记图片的名字 """
import argparse
import os
import cv2


def add_text_to_image(input_path, output_path, *,
                      font_scale=2, color=(0, 0, 255),
                      thickness=2, padding_left=25, padding_top=60
                      ):
    image = cv2.imread(input_path)
    text = os.path.basename(output_path).split(".")[0]
    cv2.putText(image, text, (padding_left, padding_top), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, color, thickness
                )
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在图片左上角添加红色文字")
    parser.add_argument("input_path", type=str, help="输入图片路径")
    parser.add_argument("output_path", type=str, help="输出图片目录")
    
    args = parser.parse_args()
    add_text_to_image(args.input_path, args.output_path)
