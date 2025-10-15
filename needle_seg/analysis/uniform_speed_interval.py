"""
寻找针梗插入皮肤时速度均匀的区间
"""
import numpy as np
from h5py.h5pl import prepend
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import cv2
import matplotlib.pyplot as plt


def ewma_smoothing(length_series, alpha=0.3):
    """
    指数加权移动平均（EWMA）平滑，适合实时处理
    
    Args:
        length_series (list): 原始数据序列
        alpha (float): 平滑系数（0 < alpha ≤ 1，越小越平滑）
    """
    smoothed_ewma = [length_series[0]]  # 初始值
    for i in range(1, len(length_series)):
        smoothed = alpha * length_series[i] + (1 - alpha) * smoothed_ewma[i - 1]
        smoothed_ewma.append(smoothed)
    return smoothed_ewma


def regular_diff(length_series):
    velocity = np.diff(length_series, prepend=length_series[0])
    acceleration = np.diff(velocity, prepend=velocity[0])
    return velocity, acceleration


def physics_aware_diff(length_series, insertion_frame):
    """
    基于物理约束的动态平滑
    假设已知插入帧，在插入前阶段强制速度为零，插入后采用自适应平滑参数。
    """
    # 插入前阶段：强制速度为零
    velocity = np.diff(length_series, prepend=length_series[0])
    velocity[:insertion_frame] = 0
    
    # 插入后阶段：对速度进行二次平滑
    velocity_post = savgol_filter(velocity[insertion_frame:], window_length=9, polyorder=2)
    velocity[insertion_frame:] = velocity_post
    
    # 加速度处理同理
    acceleration = np.diff(velocity, prepend=velocity[0])
    return velocity, acceleration


def robust_diff(length_series, step=10):
    """
    计算跨step帧的差分
    非均匀差分窗口（降低噪声敏感度）。增大差分步长，避免逐帧差分带来的噪声敏感问题。
    """
    diff = np.zeros_like(length_series)
    for i in range(step, len(length_series)):
        diff[i] = length_series[i] - length_series[i - step]
    velocity = diff / step
    acceleration = np.diff(velocity, prepend=velocity[0])
    return velocity, acceleration


def kalman_filtering(length_series):
    """
    卡尔曼滤波器
    """
    # 状态转移矩阵：假设匀速模型（位置 + 速度）
    transition_matrix = [[1, 1], [0, 1]]  # x_{k+1} = x_k + v_k, v_{k+1} = v_k
    
    # 初始化卡尔曼滤波器
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=[[1, 0]],  # 仅观测位置
        initial_state_mean=[length_series[0], 0],
        em_vars=['transition_covariance', 'observation_covariance', 'initial_state_covariance']
    )
    
    # 训练滤波器参数（使用前N帧数据）
    n_train = 30
    kf = kf.em(length_series[:n_train].reshape(-1, 1))
    
    # 应用滤波
    smoothed_length_kf, _ = kf.smooth(length_series.reshape(-1, 1))
    smoothed_length_kf = smoothed_length_kf[:, 0]  # 提取位置估计
    return smoothed_length_kf


def get_mask_length_series(video_path):
    """获取掩码视频中针梗的长度序列"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        exit()
    
    # 存储每一帧的掩码长度
    mask_lengths = []
    
    # 读取视频帧并处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 非黑色区域是掩码
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 如果没有找到轮廓，则使用上一帧的长度
        if len(contours) == 0:
            mask_lengths.append(mask_lengths[-1] if len(mask_lengths) > 0 else 0)
            continue
        
        # 计算每个轮廓的面积，并找出面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 获取该最大轮廓的最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)  # 获取矩形的四个角点
        box = np.intp(box)
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        # 获取较长的边作为掩码的长度
        max_length = max(width, height)
        mask_lengths.append(max_length)
    
    return mask_lengths


def plot_data(original_data, smoothed_data, velocity, acceleration, init_pixel_len, fps, title=None):
    """
    绘制原始数据 vs 平滑数据、速度和加速度的图像
    左侧坐标轴显示像素单位，右侧坐标轴显示真实单位 (mm, mm/s, mm/s²)
    """
    # 转换因子
    factor_length = 20 / init_pixel_len  # mm per pixel
    factor_velocity = 20 * fps / init_pixel_len  # mm/s per (pixel/frame)
    factor_acceleration = 20 * (fps**2) / init_pixel_len  # mm/s² per (pixel/frame²)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # 1. 针梗长度 (像素 vs mm)
    axs[0].plot(original_data, label='Original Data', color='gray', alpha=0.6)
    axs[0].plot(smoothed_data, label='Smoothed Data', linestyle='--', linewidth=2)
    axs[0].set_ylabel('Needle Length (Pixels)')
    axs[0].legend()
    
    # 添加右侧坐标轴，并应用转换因子
    ax0_right = axs[0].twinx()
    y0_min, y0_max = axs[0].get_ylim()
    ax0_right.set_ylim(y0_min * factor_length, y0_max * factor_length)
    ax0_right.set_ylabel('Needle Length (mm)')
    
    # 2. 速度 (像素/帧 vs mm/s)
    axs[1].plot(velocity, label='Velocity', color='green')
    axs[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axs[1].set_ylabel('Velocity (Pixels/Frame)')
    
    ax1_right = axs[1].twinx()
    y1_min, y1_max = axs[1].get_ylim()
    ax1_right.set_ylim(y1_min * factor_velocity, y1_max * factor_velocity)
    ax1_right.set_ylabel('Velocity (mm/s)')
    
    # 3. 加速度 (像素/帧² vs mm/s²)
    axs[2].plot(acceleration, label='Acceleration', color='red')
    axs[2].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axs[2].set_ylabel('Acceleration (Pixels/Frame²)')
    axs[2].set_xlabel('Frame Number')
    
    ax2_right = axs[2].twinx()
    y2_min, y2_max = axs[2].get_ylim()
    ax2_right.set_ylim(y2_min * factor_acceleration, y2_max * factor_acceleration)
    ax2_right.set_ylabel('Acceleration (mm/s²)')
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def get_init_len(length_series, velocity, max_attempt=10):
    """
    获取插入帧前的初始长度，根据速度为零的帧计算平均长度，遇见不为零即结束。
    如果一开始没有速度为零的帧，则取前 `max_attempt` 帧速度为 0 的平均长度。
    """
    lens = []
    count = 0
    for i in range(1, len(velocity)):
        if velocity[i] == 0:
            lens.append(length_series[i])
        elif count > max_attempt or len(lens) > 0:
            break
        count += 1
    return sum(lens) / len(lens)


if __name__ == '__main__':
    video_num = 103
    video_path = f'workflow/cutie/video{video_num}_masks.mp4'
    mask_lengths = get_mask_length_series(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Savitzky-Golay滤波器
    window_length = 9  # 滑动窗口长度（必须为奇数）
    polyorder = 1  # 多项式阶数
    smoothed_sg = savgol_filter(mask_lengths, window_length, polyorder)
    # velocity, acceleration = regular_diff(mask_lengths)
    # plot_data(mask_lengths, smoothed_sg, velocity, acceleration, 'Savitzky-Golay Smoothing')
    velocity, acceleration = robust_diff(smoothed_sg)
    init_pixel_len = get_init_len(smoothed_sg, velocity)
    plot_data(mask_lengths, smoothed_sg, velocity, acceleration, init_pixel_len, fps,
              'Robust Differentiation of Savitzky-Golay Smoothing'
              )
    print(f"初始长度：{init_pixel_len:.2f} pixels")
