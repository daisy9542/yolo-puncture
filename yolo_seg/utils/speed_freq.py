import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

INIT_SHAFT_LEN = 20  # 针梗的实际长度，单位为毫米
FPS = 30


def remove_outliers(data, m=2):
    """使用中位数绝对偏差（MAD）方法移除离群值。"""
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        # 无法进行判断，返回原数据
        return data.tolist()
    modified_z_scores = 0.6745 * (data - median) / mad
    cleaned_data = data[np.abs(modified_z_scores) < m]
    return cleaned_data.tolist()


def calculate_speed_mm(lens_mm, start_frame, end_frame):
    """计算速度（mm/s），在关键帧前后各 5 帧范围内。"""
    speeds_mm_s = []
    for i in range(max(0, start_frame - 5), min(len(lens_mm) - 1, end_frame + 5)):
        for j in range(i + 1, min(len(lens_mm), end_frame + 5)):
            # 计算长度变化
            distance = lens_mm[i] - lens_mm[j]  # 毫米长度变化
            time_interval = (j - i) / FPS  # 时间间隔，单位为秒
            if time_interval > 0:
                speed = distance / time_interval  # 速度，单位为毫米/秒
                speeds_mm_s.append(speed)
    return speeds_mm_s


def compute_pixel_to_mm_ratio(lens, start_frame):
    """计算像素到毫米的比例。"""
    # 获取插入开始帧前五帧的长度
    pre_frames = lens[max(0, start_frame - 5):start_frame]
    
    if not pre_frames:
        raise ValueError(f"Not enough frames before the insertion start frame {start_frame} to calculate the ratio.")
    
    # 移除偏差较大的值
    cleaned_pre_frames = remove_outliers(pre_frames, m=2)
    
    if not cleaned_pre_frames:
        raise ValueError("Not enough data to calculate the ratio after removing outliers.")
    
    # 计算平均像素长度
    avg_pixel_length = np.mean(cleaned_pre_frames)
    ratio = INIT_SHAFT_LEN / avg_pixel_length  # mm/pixel
    
    return ratio


# 读取pkl文件并绘制直方图
for filename in os.listdir("resources/needle_lens"):
    if filename.endswith(".pkl"):
        filepath = os.path.join("resources/needle_lens", filename)
        base_filename = filename.split(".")[0]
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            lens_pixels = data["lens"]  # 列表，单位为像素
            key_frame = data["key_frame"]
            insert_start_frame, insert_spec_end_frame = key_frame
            
            if insert_start_frame is None or insert_spec_end_frame is None:
                continue
            
            try:
                # 计算像素到毫米的比例
                if insert_start_frame == 0:
                    pixel_to_mm = data[0]
                else:
                    pixel_to_mm = compute_pixel_to_mm_ratio(lens_pixels, insert_start_frame)
                
                # 将所有长度转换为毫米
                lens_mm = [length * pixel_to_mm for length in lens_pixels]
                # 计算速度（毫米/秒）
                speeds_mm_s = calculate_speed_mm(lens_mm, insert_start_frame, insert_spec_end_frame)
                # 绘制速度直方图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
                # 绘制速度直方图
                counts, bins, patches = ax1.hist(speeds_mm_s, bins=30, color='blue', alpha=0.7)
                if counts.size > 0:
                    max_count_idx = counts.argmax()
                    most_frequent_range = (bins[max_count_idx], bins[max_count_idx + 1])
                    ax1.text(0.98, 0.98, f"Key Frame: {insert_start_frame}~{insert_spec_end_frame}\n"
                                         f"Speed: {most_frequent_range[0]:.2f}~{most_frequent_range[1]:.2f} mm/s",
                             transform=ax1.transAxes,
                             fontsize=12, verticalalignment='top', horizontalalignment='right')
                else:
                    ax1.text(0.5, 0.5, "No speed data available", transform=ax1.transAxes,
                             fontsize=12, verticalalignment='center', horizontalalignment='center')
                ax1.set_title(
                    f'Speed Histogram (Frames {insert_start_frame} - {insert_spec_end_frame} / Total {len(lens_mm)})')
                ax1.set_xlabel('Speed (mm/s)')
                ax1.set_ylabel('Frequency')
                ax1.grid(True)
                
                # 绘制长度变化曲线图
                ax2.plot(lens_mm, color='green')
                ax2.axvline(x=insert_start_frame, color="b", linestyle="--", label='Insert Start Frame')
                ax2.axvline(x=insert_spec_end_frame, color="b", linestyle="--", label='Insert End Frame')
                ax2.set_title("Needle Length Over Frames (mm)")
                ax2.set_xlabel("Frame Number")
                ax2.set_ylabel("Length (mm)")
                ax2.legend()
                ax2.grid(True)
                
                output_dir = 'resources/histograms'
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"{base_filename}.png"))
                plt.close()
                
                print(f"已保存直方图到 resources/histograms/{base_filename}.png")
            
            except ValueError as ve:
                print(f"Error processing {filename}: {ve}")
            except Exception as e:
                print(f"An unknown error occurred while processing {filename}: {e}")
