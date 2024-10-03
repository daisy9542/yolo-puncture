from ultralytics import YOLO
from utils.config import get_config

CONFIG = get_config()

if __name__ == '__main__':
    # Load a model
    model = YOLO(f"{CONFIG.PATH.WEIGHTS_PATH}/seg/yolov8x-seg.pt")  # Load a pretrained model (recommended for training)
    
    # Train the model
    results = model.train(data=f"{CONFIG.PATH.DATASETS_PATH}/needle-seg/data.yaml", epochs=100, imgsz=640)
    
    # Windows 使用
    # `workers=0` 单进程运行避免多进程间共享 Tensor 的动态链接库加载 shm.dll 出错
    # results = model.train(data=f"{CONFIG.PATH.DATASETS_PATH}/needle-seg/data.yaml", epochs=100, imgsz=640, workers=0)
