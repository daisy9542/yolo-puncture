import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, medfilt


def gaussian_smoothing(lens, sigma=1):
    """高斯平滑"""
    return gaussian_filter1d(lens, sigma=sigma).tolist()


def savitzky_golay_smoothing(lens, window_size=7, poly_order=3):
    """Savitzky-Golay平滑"""
    return savgol_filter(lens, window_size, poly_order).tolist()


def median_filtering(lens, kernel_size=3):
    """中值滤波"""
    return medfilt(lens, kernel_size=kernel_size).tolist()


def plot_speeds(lens, pred_range, act_range, file_path=None, frame_bias=20):
    predict_start, predict_end = pred_range
    actual_start, actual_end = act_range
    
    def plot_sub_img(ax, array, start=0, end=-1, title="", x_label="Frame", y_label="Length"):
        x_values = np.arange(0, len(lens))
        ax.plot(x_values[start:end], array[start:end])
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
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(12, 16))
    
    plot_sub_img(ax1, lens, title="Shaft Pixel Length")
    plot_sub_img(ax2, lens, start_idx, end_idx, title="Shaft Pixel Length (Range)")
    plot_sub_img(ax3, gaussian_smoothing(lens), title="Shaft Pixel Length (Gaussian Smooth)")
    plot_sub_img(ax4, gaussian_smoothing(lens), start_idx, end_idx, title="Shaft Pixel Length (Gaussian Smooth Range)")
    plot_sub_img(ax5, savitzky_golay_smoothing(lens), title="Shaft Pixel Length (Savitzky-Golay Smooth)")
    plot_sub_img(ax6, savitzky_golay_smoothing(lens), start_idx, end_idx,
                 title="Shaft Pixel Length (Savitzky-Golay Smooth Range)")
    plot_sub_img(ax7, median_filtering(lens), title="Shaft Pixel Length (Median Filter)")
    plot_sub_img(ax8, median_filtering(lens), start_idx, end_idx, title="Shaft Pixel Length (Median Filter Range)")
    
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
    
    def compute_metrics(adjust_lens):
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
    
    normal_deviation = compute_metrics(lens)
    gaussian_deviation = compute_metrics(gaussian_smoothing(lens))
    savitzky_golay_deviation = compute_metrics(savitzky_golay_smoothing(lens))
    median_filter_deviation = compute_metrics(median_filtering(lens))
    
    return normal_deviation, gaussian_deviation, savitzky_golay_deviation, median_filter_deviation
