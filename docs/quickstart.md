# 穿刺检测

本项目基于 YOLO 模型进行微调，使用特定数据集进行实例分割。

## Quick Start

### 1. 安装依赖

运行如下命令安装依赖：

```bash
python -m pip install .
```

### 2. 安装 Ultralytics

安装 Ultralytics 脚本：

```bash
python -m pip install -e .
```

### 3. 安装 mmcv

可参考[官方教程](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

MMCV 有两个版本：

- **mmcv**: 完整版，包含所有的特性以及丰富的开箱即用的 CUDA 算子。注意完整版本可能需要更长时间来编译。
- **mmcv-lite**: 精简版，不包含 CUDA 算子但包含其余所有特性和功能，类似 MMCV 1.0 之前的版本。如果不需要使用 CUDA 算子可以考虑此版本。

**注意**: 请不要在同一个环境中安装两个版本，否则可能会遇到类似 `ModuleNotFound` 的错误。在安装一个版本之前，需要先卸载另一个。`如果 CUDA 可用，强烈推荐安装 mmcv`。

在安装 mmcv 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://github.com/pytorch/pytorch#installation)。如果你使用的是搭载 apple silicon 的 mac 设备，请安装 PyTorch 1.13+ 的版本。

安装 mmcv 的命令如下：

```bash
pip install -U openmim
mim install mmcv
```

如果需要指定 mmcv 的版本，可以使用以下命令

```bash
mim install mmcv==2.0.0
```

安装 mmcv-lite

```bash
pip install -U openmim
mim install mmcv-lite
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
