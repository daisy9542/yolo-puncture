import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def gaussian_smoothing(lens, sigma=3):
    """高斯平滑"""
    return gaussian_filter1d(lens, sigma=sigma).tolist()


def savitzky_golay_smoothing(lens, window_size=7, poly_order=2):
    """Savitzky-Golay平滑"""
    return savgol_filter(lens, window_size, poly_order).tolist()


def difference(lens):
    """差分方法实现一阶导数近似"""
    # 前向差分
    diff = [(lens[i + 1] - lens[i]) for i in range(len(lens) - 1)]
    
    # 中央差分
    # diff = [(lens[i + 1] - lens[i - 1]) / 2 for i in range(1, len(lens) - 1)]
    
    diff = np.interp(np.arange(len(lens)), np.arange(len(diff)), diff)
    return diff


def plot_speeds(lens, pred_range, act_range, file_path=None, frame_bias=20):
    predict_start, predict_end = pred_range
    actual_start, actual_end = act_range
    
    def plot_sub_img(ax, array, start=0, end=-1, title="", x_label="Frame", y_label="Length"):
        x_values = np.arange(0, len(lens))
        ax.plot(x_values[start:end], array[start:end])
        if actual_start >= 0 and actual_end >= 0:
            ax.axvline(x=actual_start, color="b", linestyle="--", alpha=0.5)
            ax.axvline(x=actual_end, color="b", linestyle="--", alpha=0.5)
            ax.plot([], [], color="b", linestyle="--", label="Actual")
        ax.axvline(x=predict_start, color="g", linestyle="solid", alpha=0.5)
        ax.axvline(x=predict_end, color="g", linestyle="solid", alpha=0.5)
        ax.plot([], [], color="g", linestyle="solid", label="Predict")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
    
    start_idx = max(0, (min(predict_start, actual_start) - frame_bias))
    end_idx = min(len(lens), max(predict_start, actual_end) + frame_bias + 1)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    smooth_lens = gaussian_smoothing(lens)
    first_derivative_lens = difference(smooth_lens)
    plot_sub_img(ax1, smooth_lens, title="Shaft Pixel Length (Gaussian Smooth)")
    plot_sub_img(ax2, smooth_lens, start_idx, end_idx, title="Shaft Pixel Length (Gaussian Smooth Range)")
    plot_sub_img(ax3, first_derivative_lens, title="First Derivative", y_label="Value")
    plot_sub_img(ax4, first_derivative_lens, start_idx, end_idx, title="First Derivative Range", y_label="Value")
    
    # 保存并关闭图像
    plt.tight_layout()
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)
    plt.close()


def compute_metrics(lens, pred_range, act_range, fps, bias=5):
    predict_start, predict_end = pred_range
    actual_start, actual_end = act_range
    actual_speed = (2 * fps) / (actual_end - actual_start)
    
    def _compute_metrics(adjust_lens):
        pixel_lens = adjust_lens[max(0, predict_start - bias): max(1, predict_start)]
        avg_pixel_len = sum(pixel_lens) / len(pixel_lens)
        adjust_predict_end = predict_start + 1
        for i in range(predict_start + 1, len(adjust_lens)):
            if adjust_lens[i] <= avg_pixel_len * 0.9:
                adjust_predict_end = i
                break
        speed = (2 * fps) / (adjust_predict_end - predict_start)
        deviation = abs((actual_speed - speed) / actual_speed)
        return deviation
    
    normal_deviation = _compute_metrics(lens)
    gaussian_deviation = _compute_metrics(gaussian_smoothing(lens))
    savitzky_golay_deviation = _compute_metrics(savitzky_golay_smoothing(lens))
    
    return normal_deviation, gaussian_deviation, savitzky_golay_deviation
