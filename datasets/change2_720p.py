"""
将 VIPSeg 格式数据集分辨率转化为 720P
"""
import os
from PIL import Image
from multiprocessing import Pool

ROOT = 'resources/datasets/needle-seg-full'
DIR = ROOT + '/images'
DIR2 = ROOT + '/panomasks'
Target_Dir = ROOT + '/VIPSeg_720P'


def change(DIR, video, image):
    # 判断图像和掩码文件是否存在
    if os.path.isfile(os.path.join(Target_Dir, 'images', video, image)) and os.path.isfile(os.path.join(Target_Dir, 'panomasks', video, image.split('.')[0] + '.png')):
        return

    img = Image.open(os.path.join(DIR, video, image))
    w, h = img.size
    img = img.resize((int(720 * w / h), 720), Image.BILINEAR)

    # 检查掩码文件是否存在
    if not os.path.isfile(os.path.join(DIR2, video, image.split('.')[0] + '.png')):
        print('This is the test set')
        print(os.path.join(DIR2, video, image.split('.')[0] + '.png'))
        return

    mask = Image.open(os.path.join(DIR2, video, image.split('.')[0] + '.png'))
    mask = mask.resize((int(720 * w / h), 720), Image.NEAREST)

    # 确保目标文件夹存在
    if not os.path.exists(os.path.join(Target_Dir, 'images', video)):
        os.makedirs(os.path.join(Target_Dir, 'images', video))
    if not os.path.exists(os.path.join(Target_Dir, 'panomasks', video)):
        os.makedirs(os.path.join(Target_Dir, 'panomasks', video))

    # 保存处理后的图像和掩码
    img.save(os.path.join(Target_Dir, 'images', video, image))
    mask.save(os.path.join(Target_Dir, 'panomasks', video, image.split('.')[0] + '.png'))
    print(f'Processing video {video} image {image}')


def process_videos(DIR):
    for video in sorted(os.listdir(DIR)):
        if video[0] == '.':  # 跳过隐藏文件夹
            continue
        for image in sorted(os.listdir(os.path.join(DIR, video))):
            if image[0] == '.':  # 跳过隐藏文件
                continue
            # 调用 change 函数处理每个视频和图像
            change(DIR, video, image)


if __name__ == '__main__':
    # 使用多进程池来加速处理
    p = Pool(28)
    p.apply_async(process_videos, args=(DIR,))
    p.close()
    p.join()

    print('Finish')
