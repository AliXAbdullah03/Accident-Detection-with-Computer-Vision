from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch

# Use the model
model.train(data="E:/Backend-Modules-main\Vehicle detection.v4i.yolov8/data.yaml", epochs=50)  # train the model
