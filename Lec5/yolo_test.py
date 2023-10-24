from ultralytics import YOLO

model = YOLO('models/best.pt')

model.predict('test_pics/construction-safety.jpg', save=True)
