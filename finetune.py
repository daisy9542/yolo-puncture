from ultralytics import YOLO

model = YOLO('yolov10x.pt')

model.train(data='datasets/needle/data.yaml', epochs=200, batch=32, imgsz=640, device='0,1', freeze=10) # 大概23个layers, freeze不能超过23
