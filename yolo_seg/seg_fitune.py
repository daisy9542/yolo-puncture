import torch.cuda

from ultralytics import YOLO
from utils.config import get_config

CONFIG = get_config()

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Load a model
    model = YOLO(f"{CONFIG.PATH.WEIGHTS_PATH}/seg/yolo11n-seg.pt")  # Load a pretrained model (recommended for training)
    
    # Train the model
    results = model.train(data=f"{CONFIG.PATH.DATASETS_PATH}/needle-seg/data.yaml",
                          epochs=100,
                          imgsz=1280,
                          device=device)
    
    # Windows 使用
    # `workers=0` 单进程运行避免多进程间共享 Tensor 的动态链接库加载 shm.dll 出错
    # results = model.train(data=f"{CONFIG.PATH.DATASETS_PATH}/needle-seg/data.yaml",
    #                       epochs=100,
    #                       imgsz=1280,
    #                       workers=0,
    #                       device = device)
