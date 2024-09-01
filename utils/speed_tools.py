import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, medfilt


def gaussian_smoothing(lens, sigma=3):
    """高斯平滑"""
    return gaussian_filter1d(lens, sigma=sigma).tolist()


def savitzky_golay_smoothing(lens, window_size=7, poly_order=2):
    """Savitzky-Golay平滑"""
    return savgol_filter(lens, window_size, poly_order).tolist()


def median_filtering(lens, kernel_size=7):
    """中值滤波"""
    return medfilt(lens, kernel_size=kernel_size).tolist()


def first_derivative(lens):
    res = [0]
    res.extend([(lens[i + 1] - lens[i]) for i in range(len(lens) - 1)])
    return res


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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    smooth_lens = gaussian_smoothing(lens)
    first_derivative_lens = first_derivative(smooth_lens)
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
    median_filter_deviation = _compute_metrics(median_filtering(lens))
    
    return normal_deviation, gaussian_deviation, savitzky_golay_deviation, median_filter_deviation


if __name__ == '__main__':
    speeds = [225.42832946777344, 229.38540649414062, 225.5670623779297, 225.42835998535156, 225.42832946777344,
              228.8287353515625, 225.42666625976562, 226.0478515625, 222.80479431152344, 219.65225219726562,
              219.53079223632812, 218.29830932617188, 218.56283569335938, 215.78614807128906, 217.7197723388672,
              213.21214294433594, 217.203857421875, 216.95594787597656, 214.39358520507812, 214.086181640625,
              217.24729919433594, 214.4048614501953, 212.90090942382812, 211.9661102294922, 214.78179931640625,
              213.01280212402344, 214.7854766845703, 213.36602783203125, 215.23867797851562, 213.83416748046875,
              214.7855224609375, 216.17498779296875, 214.77638244628906, 214.78079223632812, 212.8978729248047,
              214.77052307128906, 215.71360778808594, 213.8291015625, 215.69723510742188, 213.8197021484375,
              212.8978729248047, 212.8978729248047, 209.62069702148438, 206.8191375732422, 211.02749633789062,
              208.20692443847656, 205.9001922607422, 206.82281494140625, 203.0709686279297, 199.32281494140625,
              199.32164001464844, 199.80996704101562, 198.03656005859375, 199.32281494140625, 195.5731964111328,
              195.1427001953125, 193.25572204589844, 194.18634033203125, 189.10784912109375, 190.0737762451172,
              192.31744384765625, 188.16189575195312, 189.10769653320312, 189.1014404296875, 184.0445556640625,
              184.36105346679688, 185.27694702148438, 179.93276977539062, 179.93194580078125, 180.23939514160156,
              176.1573944091797, 178.06898498535156, 175.2041473388672, 175.204345703125, 175.20167541503906,
              174.03558349609375, 171.1634979248047, 169.0740509033203, 169.09530639648438, 169.36557006835938,
              169.08843994140625, 164.2010040283203, 168.0718536376953, 163.24554443359375, 159.35491943359375,
              157.45140075683594, 155.23175048828125, 153.5205078125, 151.57374572753906, 151.57778930664062,
              151.34222412109375, 148.22044372558594, 151.6471710205078, 149.9130401611328, 147.99884033203125,
              149.89776611328125, 151.159423828125, 145.82369995117188, 146.45457458496094, 148.7524871826172,
              146.72784423828125, 142.9717559814453, 145.7772674560547, 145.7540740966797, 145.773681640625,
              144.91839599609375, 146.1004638671875, 143.9613037109375, 141.20388793945312, 140.02378845214844,
              137.2868194580078, 140.02691650390625, 133.36392211914062, 133.19338989257812, 133.93075561523438,
              131.23953247070312, 130.23684692382812, 125.15384674072266, 126.3610610961914, 119.57005310058594,
              119.56928253173828, 115.69351959228516, 112.78614807128906, 107.9674301147461, 102.08723449707031,
              98.21279907226562, 95.32823181152344, 90.52059936523438, 86.66309356689453, 81.74346923828125,
              77.86387634277344, 73.00322723388672, 69.85025787353516, 69.1226577758789, 64.28935241699219,
              66.19236755371094, 59.421226501464844, 56.51545715332031, 51.64836883544922, 45.839229583740234,
              42.928802490234375, 40.01837158203125, 39.04823303222656, 36.13718032836914, 36.09904479980469,
              33.23170852661133, 32.27592849731445, 28.407306671142578, 28.144268035888672, 25.0, 23.525955200195312,
              21.0, 22.0, 22.0, 22.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
    x = speeds[19:24]
    print(x, sum(x) / len(x))
    print(speeds[25: 50])
    print(compute_metrics(speeds, (24, 58), (29, 57), 30))
    plot_speeds(speeds, (24, 58), (29, 57))
