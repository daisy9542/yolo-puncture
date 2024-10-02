from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # Load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/puncture/datasets/needle-seg/data.yaml", epochs=100, imgsz=640)
