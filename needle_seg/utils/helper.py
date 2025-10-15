def parse_video_range(value: str) -> list[int]:
    """
    解析视频编号支持单个或范围形式：
    - 102          => [102]
    - 101-105      => [101, 102, 103, 104, 105]
    - 1,3,5-7,9    => [1, 3, 5, 6, 7, 9]
    """
    result = []
    parts = value.split(',')

    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    return result