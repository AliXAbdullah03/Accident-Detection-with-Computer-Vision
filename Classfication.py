from ultralytics import YOLO
VIDEOS_DIR = "videos\classification"
model_path = 'yolov8_pretrained models\yolov8n.pt'
frame = "videos\\classification_videos\\video1.mp4"  
model = YOLO(model_path)
results = model(frame, show=False, conf=0.5, save=True)







