from PIL import Image, ImageDraw
import os
import glob

def apply_normalized_polygon_mask(image_path, normalized_polygon, output_path):
    # 打开原图
    image = Image.open(image_path)
    width, height = image.size

    # 将归一化的多边形坐标转换为实际像素坐标
    polygon = [(int(x * width), int(y * height)) for x, y in normalized_polygon]

    # 创建一个与原图大小相同的黑色图像
    black_image = Image.new('L', (width, height), 0)

    # 创建一个ImageDraw对象
    draw = ImageDraw.Draw(black_image)

    # 将多边形区域填充为白色（255）
    draw.polygon(polygon, outline=255, fill=255)

    # 保存结果到输出路径
    black_image.save(output_path, 'PNG')

def read_label_txt(txt_path):
    # 打开并读取文件内容
    with open(txt_path, 'r') as file:
        data = file.read()
    # 将读取的内容分割成一个列表
    numbers = data.split()
    # 忽略第一个数字
    numbers = numbers[1:]
    # 将数据转换为浮点数
    float_numbers = [float(num) for num in numbers]
    # 将浮点数成对放入元组，并存入列表
    pairs = [(float_numbers[i], float_numbers[i + 1]) for i in range(0, len(float_numbers), 2)]

    return pairs

if __name__ == '__main__':
    mode = 'train'
    image_dir = f'/home/puncture/datasets/needle-seg/images/{mode}/'  # 原图路径
    label_dir = f'/home/puncture/datasets/needle-seg/labels/{mode}/'  # 标签txt路径
    output_dir = f'/home/puncture/datasets/needle-seg/labels_mask/{mode}/'  # 输出图片路径

    os.makedirs(output_dir, exist_ok=True)

    txt_files = glob.glob(os.path.join(label_dir, '*.txt'))
    for file in txt_files:
        fname = os.path.splitext(os.path.basename(file))[0] 
        normalized_polygon = read_label_txt(os.path.join(label_dir, f'{fname}.txt'))
        apply_normalized_polygon_mask(os.path.join(image_dir, f'{fname}.jpg'), normalized_polygon, os.path.join(output_dir, f'{fname}.png'))

    