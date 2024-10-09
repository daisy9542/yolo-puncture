# 穿刺检测

本项目基于 YOLO 模型进行微调，使用特定数据集进行实例分割。

## Quick Start

### 1. 安装依赖

运行如下命令安装依赖：

```bash
python -m pip install .
```

### 2. 安装 Ultralytics

安装改进后的 Ultralytics 依赖：

```bash
python -m pip install -e ultralytics
```

### 3. 安装子模块

安装子模块 DEVA、Segment Anything 和 thinplate。

> thinplate 是模块 DEVA 的依赖，如无法从 GitHub 获取，则可以从本地安装。
> 同时，将 deva/pyproject.toml 中对 thinplate 的依赖需求注释掉。

```bash
git submodule update --init --recursive
```

## 微调 YOLO 模型

> [YOLO CLI](https://docs.ultralytics.com/zh/usage/cli)

Windows 使用 `workers=0` 单进程运行避免多进程间共享 Tensor 的动态链接库加载 shm.dll 出错。

在 `yolo11n-seg.pt` 上进行微调：

```bash
yolo train data=/home/puncture/datasets/needle-seg/data.yaml model=yolo11n-seg.pt epochs=100 imgsz=1280 device=cuda
```

在 `yolo11x-seg.pt` 上进行微调：

```bash
yolo train data=/home/puncture/datasets/needle-seg/data.yaml model=yolo11x-seg.pt epochs=100 imgsz=1280 batch=8 device=cuda
```
