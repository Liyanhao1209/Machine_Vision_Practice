from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='config/safehat.yaml', epochs=1)

model.val()
