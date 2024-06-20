from ultralytics import YOLOv10

model = YOLOv10.from_pretrained('weights/yolov10m/', local_files_only=True)

model.train(data='datasets/needle/data.yaml', epochs=500, batch=64, imgsz=640, device='2,3')
