import argparse


def calculate_speed(start_frame, end_frame_min, end_frame_max, frame_rate=30, distance_mm=2):
    # 计算最小和最大情况的帧数差
    frame_count_max = end_frame_min - start_frame
    frame_count_min = end_frame_max - start_frame
    
    # 计算时间（秒）
    time_min = frame_count_min / frame_rate
    time_max = frame_count_max / frame_rate
    
    # 计算平均速度（毫米/秒）
    speed_min = distance_mm / time_min
    speed_max = distance_mm / time_max
    
    # 打印结果
    print(f"min: {speed_min:.2f} mm/s")
    print(f"max: {speed_max:.2f} mm/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calc tools")
    parser.add_argument("--start_frame", type=int, required=True, help="开始帧")
    parser.add_argument("--end_frame_min", type=int, required=True, help="结束帧最小值")
    parser.add_argument("--end_frame_max", type=int, required=True, help="结束帧最大值")
    parser.add_argument("--frame_rate", type=int, default=30, help="帧率（默认 30 帧）")
    parser.add_argument("--distance_mm", type=int, default=2, help="移动距离（默认 2 毫米）")
    
    args = parser.parse_args()
    
    calculate_speed(args.start_frame, args.end_frame_min, args.end_frame_max, args.frame_rate, args.distance_mm)