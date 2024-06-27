# 定义帧率和移动距离
frame_rate = 30  # 帧率：30帧/秒
distance_mm = 2  # 移动距离：2毫米

# 定义起始帧和结束帧的范围
start_frame_min = 1
start_frame_max = 3
end_frame_min = 25
end_frame_max = 29

# 计算最小和最大情况的帧数差
frame_count_min = end_frame_min - start_frame_min
frame_count_max = end_frame_max - start_frame_max

# 计算时间（秒）
time_min = frame_count_min / frame_rate
time_max = frame_count_max / frame_rate

# 计算平均速度（毫米/秒）
speed_min = distance_mm / time_min
speed_max = distance_mm / time_max

# 打印结果
print(f"min: {speed_min:.2f} mm/s")
print(f"max: {speed_max:.2f} mm/s")
