from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # or yolo8s.pt, depending on what you have installed

# Train the model
result = model.train(
    data="/home/ws/datasets/YOLODataset/dataset.yaml",
    imgsz=960,
    epochs=200,
    project="/home/ws/detect",
    name="yolov8n-custom",
    exist_ok=True,
)
